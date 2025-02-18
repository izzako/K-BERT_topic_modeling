import numpy as np
import re
import pandas as pd
import nltk
from pathlib import Path
import os
import emoji
nltk.download('stopwords')
nltk.download('punkt')
from time import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import collections
import plotly
import plotly.express as px
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import pickle
from cuml.cluster import KMeans
import kneed
import cudf
from cuml.preprocessing import RobustScaler
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from datetime import date
from cuml.cluster.hdbscan import HDBSCAN
from cuml import UMAP, DBSCAN, TSNE
import matplotlib.pyplot as plt
import json
import sys

from traceback import print_exc
# from tenacity import (
#     retry,
#     stop_after_attempt,
#     wait_random_exponential,
# )

# from openai import OpenAI


class topic_modelling():
    def __init__(self,X,data):

        # self.sources = sources
        # self.source_title = ', '.join([source.title() for source in sources])
        self.data = data#[data[kw]['source'].isin(sources)].copy()
        self.X = X#[self.data.index.to_numpy(),:]
        self.data.reset_index(drop=True,inplace=True)
    
    def get_cluster(self,kmax=50):
        print('Determining topic number')
        sse = []
        X_cudf= cudf.DataFrame(self.X) #convert to cudf
        for k in tqdm(range(1, kmax+1)): #iterate over K
            kmeans = KMeans(n_clusters=k,random_state=24).fit(X_cudf)
            centroids = kmeans.cluster_centers_.to_cupy().get()
            pred_clusters = kmeans.labels_.to_cupy().get()
            curr_sse = 0
            
            # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
            for i in range(len(self.X)):
                curr_center = centroids[pred_clusters[i]]
                curr_sse += np.linalg.norm(self.X[i,:] - curr_center)
            
            sse.append(curr_sse)
        self.kneedle = kneed.KneeLocator(np.arange(kmax)+1, sse, S=1.0, curve="convex", direction="decreasing")
        self.topic_num = self.kneedle.elbow #get elbow point
        print('Number of Topic:',self.topic_num)
        
        print('Clustering...')
        self.clustering_model = KMeans(n_clusters=self.topic_num,random_state=24) # get clustering with ideal k
        cols=['id','date','country','city','source',
            'page_account_name','gender','url','text','cleaned_text','cluster_id',
            'dist']
        self.clustering_model.fit(X_cudf)
        cluster_ids_x = self.clustering_model.labels_.to_cupy().get()
        self.cluster_centers = self.clustering_model.cluster_centers_.to_cupy().get()
        self.data['cluster_id'] = np.array(cluster_ids_x)
        dist = []
        for i in tqdm(np.arange(len(self.data))):
            dist.append(np.linalg.norm(self.X[i,:]-np.array(self.cluster_centers[self.data.loc[i,'cluster_id']])))
        self.data['dist'] = dist

    def save_data(self): #use this after get_cluster() for reproducible results!!
        if not os.path.exists(os.path.join(BASE_PATH,'predicted data')):
            os.mkdir(os.path.join(BASE_PATH,'predicted data'))
        date_today = date.today().strftime("%Y%m%d")
        self.data.to_csv(os.path.join(BASE_PATH,'predicted data',f'kmeans-{self.kw}-{date_today}.csv'),index=False,sep=';',quoting=1)
        print('data saved in',os.path.join(BASE_PATH,'predicted data',f'kmeans-{self.kw}-{date_today}.csv'))

    def visualize_cluster2(self,params,sources=None,size=10,topic_label=None):
        if not sources:
            sources = ['Facebook','Instagram','twitter','news']
            source_title = 'all'
        else:
            source_title = ', '.join([source.title() for source in sources])
        X_source = self.X[self.data[self.data['source'].isin(sources)].index.to_numpy(),:]
        print('reducing dimention...')
        X_2d= UMAP(**params).fit_transform(self.X)
        labels_ = self.data['cluster_id']
        if topic_label:
            labels_ = [topic_label[i] for i in labels_]
        else:
            labels_ = [str(i) for i in labels_]
        #kmeans
        print('plotting...')
        fig = px.scatter(x=X_2d[:,0],y=X_2d[:,1],color=labels_,width=600,height=500,#,color_continuous_scale='turbo'
                    title=f"{self.scrape_keywords} topic modelling <br> on {source_title} data",
                    color_discrete_sequence=plotly.colors.qualitative.Dark24)
        fig.update_traces(marker=dict(size=1))
        fig.update_layout(title_x=0.5,title_y=0.97)
        fig.update_yaxes(range = [-1*size,size],constrain='domain',title='',showticklabels=False)
        fig.update_xaxes(range = [-1*size,size],scaleanchor= 'x',title='',showticklabels=False)
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',margin=dict(l=10,r=10,t=40,b=10))
        fig.update_layout(coloraxis_colorbar=dict(
            title="Topic",
        ),)
        fig.update_layout(legend= {'itemsizing': 'constant','title':'Topic'})
        return fig

    def get_keywords2(self,sources=None):
        if not sources:
            sources = ['Facebook','Instagram','twitter','news']
        data_filtered = self.data[self.data['source'].isin(sources)]
        print('getting the keywords..')
        cluster_kw = {}
        data_filtered['dist_rank'] = data_filtered.groupby('cluster_id')['dist'].rank(method='first')
        for i in tqdm(range(data_filtered.cluster_id.nunique())):
            cluster_sent = [ ' '.join([re.sub('[^a-zA-Z]','',w).lower() for w in word_tokenize(text) if (w.lower() not in list_stopwords) and (len(w)>1)]) for text in data_filtered.loc[(data_filtered.dist_rank<=100)&(data_filtered.cluster_id==i),'cleaned_text'].to_list()]
            stemmed_list =  stemmer.stem(' '.join(cluster_sent)).split()
            bow = (collections.Counter([i for i in stemmed_list if i not in list_stopwords.union(post_stem_stopwords)])).items()
            cluster_kw[i] = pd.DataFrame(bow,columns=['text','freq']).sort_values(by='freq',ascending=False).iloc[:10,0].to_list()
        return cluster_kw