import re
import emoji

def cleaning_text(df,text_col):

    re_dict = {
        r'\n':' ',
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)':'HTTPURL',
        r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})':'HTTPURL',
        r"\@\w+":'@USER',
        '://':'',
        r'#\w+':'',
        r'\,+':r',',
        r'\.+':r'.',
        r'\xa0':' ',
        r"\”":r'\"',
        r"\\":'',
        r"&nbsp;":" ",
        r' +':' ',
    }

    df[text_col] = df[text_col].astype(str)
    df.loc[:,'cleaned_text'] = df[text_col].apply(emoji.demojize)
    df.loc[:,'cleaned_text'] = df['cleaned_text'].apply(lambda x: x.encode('utf-8').decode('ascii','ignore'))
    df.drop_duplicates('cleaned_text',inplace=True)
    df.loc[:,'cleaned_text'] = df['cleaned_text'].replace(re_dict,regex=True).str.strip()
    df.reset_index(drop=True,inplace=True)
    return df


def neutralize_ads(dataf,text_col):
  for i in ['#openBO','#partnerpasutri','#JudiOnline','Slot Gacor',
            'bokep','colmek','ngewe','crot','sange',
            '#pijat[a-z]+','#gigolo[a-z]+','#pasutri[a-z]+','pijit sensual',
            '#sangekberat','#viralmesum',"privasi terjamin 100%",'privasi 100%',
            'ready open','ready partner','ready pijat','ready sayang','#sangeberat']:
    dataf = dataf[~dataf[text_col].astype(str).str.contains(i,regex=True,case=False)]
  return dataf