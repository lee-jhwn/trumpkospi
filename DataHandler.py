import pandas as pd
import json
from datetime import datetime, timedelta, time
from kospi_preprocess import kospi_data

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

with open('trumptwts_17198.json') as f:
    for line in f:
        tweets = pd.DataFrame(json.loads(line))


tweets = tweets.drop(columns=['source', 'id_str', 'retweet_count', 'favorite_count'])
tweets = tweets[tweets.is_retweet == False]
tweets = tweets.drop(columns=['is_retweet'])

print('setting to seoul time, GMT +9 ...')
for i,v in tweets.iterrows():
    v['created_at'] = set_seoul_time(v['created_at'])

tweets = tweets.sort_values(['created_at'])
tweets = tweets.rename(columns={'created_at': 'date'})

tweets = tweets.merge(kospi_data, on='date', how='left')
tweets = tweets.drop(columns=['date', 'kospi'])
tweets = tweets.fillna(method='bfill').dropna()

if __name__=='__main__':

    print(tweets)

    # print(kospi_data)


