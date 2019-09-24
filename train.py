from DataHandler import get_tweets
import pickle
import tensorflow as tf
import numpy as np
from pprint import pprint

from gensim.models import KeyedVectors, FastText, Word2Vec # 미리 훈련된 단어 벡터 읽기

from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dropout, Dense, LSTM, Bidirectional
from keras.callbacks import LambdaCallback

# from keras.models import Model
# from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Concatenate, Dropout, Dense, LSTM
from keras.constraints import max_norm

try:
    with open('tweets.pkl', 'rb') as f:
        tweets = pickle.load(f)
    print('loading saved tweets...')

except:

    tweets = get_tweets()
    with open('tweets.pkl', 'wb') as f:
        pickle.dump(tweets,f)

train_cut = 0.8
train_sentences = list(tweets.iloc[:int(train_cut*len(tweets)), 0])
train_label = list(tweets.iloc[:int(train_cut*len(tweets)), 1])
test_sentences = list(tweets.iloc[int(train_cut*len(tweets)):, 0])
test_label = list(tweets.iloc[int(train_cut*len(tweets)):, 1])
# pprint(train_sentences[:5])

w2v_size = 100

print('embedding pretrain...')
if False:
    w2v_model = FastText([['<dummy>']] + train_sentences + test_sentences, sg=1, min_count=1, sorted_vocab=False, size=w2v_size)
    with open('embedding_pretrain.pkl', 'wb') as f:
        pickle.dump(w2v_model, f)
else:
    with open('embedding_pretrain.pkl', 'rb') as f:
        w2v_model = pickle.load(f)

# print(w2v_model.wv.vocab['<dummy>'].index) #dummy index 0인지 확인
pretrained_weights = w2v_model.wv.vectors
vocab_size = pretrained_weights.shape[0]
print(f'단어 임베딩 형상: {pretrained_weights.shape}')

max_length = max([len(sentence) for sentence in train_sentences + test_sentences])
print(max_length)
mean = np.mean([len(sentence) for sentence in train_sentences + test_sentences])
median = np.median([len(sentence) for sentence in train_sentences + test_sentences])
theta = 1 # 값 조절하며 maxlen 정하기 1: maxlen, 0: median

maxlen = int(max_length * theta + median * (1-theta))
print('final maxlen:', maxlen)


def get_image(sentence):
    image = np.zeros(maxlen, dtype=np.int32)
    for i, word in enumerate(sentence):
        if i >= maxlen:
            break
        image[i] = w2v_model.wv.vocab[word].index

    return image

def get_corr(x, y):
    return np.correlate(x, y)

print(len(train_sentences))
print(len(test_sentences))

X_train = [get_image(sentence) for sentence in train_sentences]
X_test = [get_image(sentence) for sentence in test_sentences]

X_train = np.stack(X_train, axis=0)
X_test = np.stack(X_test, axis=0)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=w2v_size, mask_zero=True, weights=[pretrained_weights], trainable=False))
model.add(Bidirectional(LSTM(units=128, return_sequences=False, dropout=0.3)))
# model.add(LSTM(units=128, return_sequences=False, dropout=0.5))
model.add(Dense(units=32, activation='tanh')) # unit, activation 바꿔보기
# model.add(Dense(units=8, activation='elu')) # unit, activation 바꿔보기
model.add(Dense(units=1, activation='tanh'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()

model.fit(X_train, train_label, validation_data=[X_test, test_label], epochs=60, batch_size=128, verbose=2, callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])






# for i,v in data.iterrows():
#     v['percent']

