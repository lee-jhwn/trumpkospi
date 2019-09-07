from DataHandler import tweets
import pickle

try:
    with open('tweets.pkl', 'rb') as f:
        data = pickle.load(f)
    print('loading saved tweets...')

except:

    data = tweets
    with open('tweets.pkl', 'wb') as f:
        pickle.dump(tweets,f)


print(data)