import pandas as pd
import json
from datetime import datetime, timedelta, time
from kospi_preprocess import kospi_data
import re
import os
import pickle
# from gensim.models import KeyedVectors
from nltk import word_tokenize


# http://www.trumptwitterarchive.com/
# trump tweet archive
# 20170120 ~ 20190831
# exclude RT
# set to seoul time


def set_seoul_time(t):

    t = t[4:19] + t[25:]
    t = datetime.strptime(t, '%b %d %H:%M:%S %Y') + timedelta(hours=9)
    if t.time() < time(hour=15,minute=0,second=0): # tweets after 3pm -> consider next day
        t = t.date()
    else:
        t = t + timedelta(days=1)
        t = t.date()
    return str(t)
#
# with open('condensed_2017.json') as f:
#     for line in f:
#         data_2017 = pd.DataFrame(json.loads(line))
#
# with open('condensed_2018.json') as f:
#     for line in f:
#         data_2018 = pd.DataFrame(json.loads(line))
#
# tweets = pd.concat([data_2017, data_2018])

def get_tweets():
    # os.system('wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"')

    with open('trumptwts_17198.json') as f:
        for line in f:
            tweets = pd.DataFrame(json.loads(line))


    tweets = tweets.drop(columns=['source', 'id_str', 'retweet_count', 'favorite_count'])
    tweets = tweets[tweets.is_retweet == False]
    tweets = tweets.drop(columns=['is_retweet'])

    print('setting to seoul time, GMT +9 ...')
    for i,v in tweets.iterrows():
        v['created_at'] = set_seoul_time(v['created_at'])
        v['text'] = word_tokenize((v['text'].lower()))

    tweets = tweets.sort_values(['created_at'])
    tweets = tweets.rename(columns={'created_at': 'date'})

    tweets = tweets.merge(kospi_data, on='date', how='left')
    tweets = tweets.drop(columns=['date', 'kospi'])
    tweets = tweets.fillna(method='bfill').dropna()

    tweets = tweets.sample(frac=1, random_state=7).reset_index(drop=True)

    # w2v_mod = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True, limit=30000)
    # n = len(max(tweets['text']))
    # print(n)
    # k = w2v_mod.vector_size

    # image =


    return tweets


if __name__=='__main__':

    tweets = get_tweets()
    with open('tweets.pkl', 'wb') as f:
        pickle.dump(tweets,f)

    print(tweets)


    # print(kospi_data)


