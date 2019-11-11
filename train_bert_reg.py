from DataHandler import get_tweets
import pickle
# import tensorflow as tf
import numpy as np
from pprint import pprint
from config import *
import os
import codecs

from gensim.models import KeyedVectors, FastText, Word2Vec # 미리 훈련된 단어 벡터 읽기

from keras.models import Sequential, Model
from keras.layers import Embedding, Input, Dense, LSTM, Bidirectional, BatchNormalization, Concatenate, Flatten, Masking, Permute
from keras.callbacks import LambdaCallback
from keras.optimizers import Adam
from keras_attention.models import AttentionWeightedAverage
from keras.utils.training_utils import multi_gpu_model
from keras import backend as K
import heapq
from keras_bert import extract_embeddings, Tokenizer, load_trained_model_from_checkpoint
import codecs
from keras_radam import RAdam


K.tensorflow_backend._get_available_gpus()

SEQ_LEN = 128
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-4

pretrained_path = 'uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')



try:
    with open('tweets.pkl', 'rb') as f:
        tweets = pickle.load(f)
    print('loading saved tweets...')

except:

    tweets = get_tweets()
    with open('tweets.pkl', 'wb') as f:
        pickle.dump(tweets,f)

train_sentences = list(tweets.iloc[:int(train_cut*len(tweets)), 0])
train_label = list(tweets.iloc[:int(train_cut*len(tweets)), 1])
test_sentences = list(tweets.iloc[int(train_cut*len(tweets)):, 0])
test_label = list(tweets.iloc[int(train_cut*len(tweets)):, 1])
# pprint(train_sentences[:5])
del tweets









max_length = max([len(sentence) for sentence in train_sentences + test_sentences])
print(max_length)
mean = np.mean([len(sentence) for sentence in train_sentences + test_sentences])
median = np.median([len(sentence) for sentence in train_sentences + test_sentences])
theta = 1 # 값 조절하며 maxlen 정하기 1: maxlen, 0: median

maxlen = int(max_length * theta + median * (1-theta))
print('final maxlen:', maxlen)

token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

# print(token_dict)
tokenizer = Tokenizer(token_dict)

# print(train_sentences)

X_train = np.array([tokenizer.encode(first=' '.join(sentence), max_len=maxlen)[0] for sentence in train_sentences])
mod = len(X_train) % BATCH_SIZE
print(mod)
X_train = X_train[:-mod]
train_label = train_label[:-mod]
X_test = np.array([tokenizer.encode(first=' '.join(sentence), max_len=maxlen)[0] for sentence in test_sentences])
mod = len(X_test) % BATCH_SIZE
print(mod)
X_test = X_test[:-mod]
test_label = test_label[:-mod]
X_train = [X_train, np.zeros_like(X_train)]
X_test = [X_test, np.zeros_like(X_test)]

print(X_train[0].shape)
print(X_test[0].shape)

# pprint(X_train)


model = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    training=True,
    trainable=False,
    seq_len=maxlen,
)


inputs = model.inputs[:2]
print(inputs)
dense = model.get_layer('NSP-Dense').output
dense = Dense(units=128, activation='tanh')(dense)
outputs = Dense(units=1, activation='tanh')(dense)
model = Model(inputs, outputs)
model.compile(RAdam(lr=LR),loss='mean_squared_error', metrics=['mae'])
# model.summary()




def get_corr(x, y):
    return np.correlate(x, y)

# print(len(train_sentences))
# print(len(test_sentences))

# if which_embedding == 'Google_W2V':
#
#     X_train = [get_image(sentence) for sentence in train_sentences]
#     X_test = [get_image(sentence) for sentence in test_sentences]
#     X_train = np.stack(X_train, axis=0)
#     X_test = np.stack(X_test, axis=0)
#
#     print(len(pretrained_weights))
#
#     del w2v_model

#
# def get_word(idx):
#     return [key for key,value in token_dict.items() if value == idx][0]
#
# if which_embedding == 'BERT':

    # try:
    #     with open('bert_token_dict.pkl', 'rb') as f:
    #         token_dict = pickle.load(f)
    # except:
    #     token_dict = {}
    #     with codecs.open('uncased_L-12_H-768_A-12/vocab.txt', 'r', 'utf8') as reader:
    #         for line in reader:
    #             token = line.strip()
    #             token_dict[token] = len(token_dict)
    #     with open('bert_token_dict.pkl', 'wb') as f:
    #         pickle.dump(token_dict, f)

    # with codecs.open('uncased_L-12_H-768_A-12/vocab.txt', 'r', 'utf8') as reader:
    #     vocab = [line.strip() for line in reader]


    # tokenizer = Tokenizer(token_dict)
    # tokens = [tokenizer.tokenize(" ".join(sentence)) for sentence in train_sentences + test_sentences]
    # maxlen = max([len(sentence) for sentence in tokens])
    # for i,sentence in enumerate(tokens):
    #     while len(tokens[i]) < maxlen:
    #         tokens[i].append('[PAD]')

    # print(os.getcwd())
    # print(len(tokens[5]))
    # print('maxlen_bert :', maxlen)
    # indices, segments = tokenizer.encode(first=' '.join(test_sentences[0]), max_len=maxlen)
    # print(indices)
    # print(" ".join(get_word(w) for w in indices))
    # for w in indices:
    #     print
    # X_train = [tokenizer.encode(first=' '.join(sentence), max_len=maxlen)[0] for sentence in train_sentences]
    # X_test = [tokenizer.encode(first=' '.join(sentence), max_len=maxlen)[0] for sentence in test_sentences]
    # X_train = np.stack(X_train, axis=0)
    # X_test = np.stack(X_test, axis=0)
    # print(X_test)

    # texts = [get_word(idx) for idx in range(len(token_dict))]
    # print(vocab)

# try:
#     with open('bert_embedding_sent.pkl', 'rb') as f:
#         print('loading existing bert embedding...')
#         bert_embedding = pickle.load(f)
#         print('loaded')
# except:
#     print('loading bert embedding')
#     bert_embedding = extract_embeddings('uncased_L-12_H-768_A-12', [" ".join(sentence) for sentence in train_sentences + test_sentences])
#     # bert_embedding = extract_embeddings('uncased_L-12_H-768_A-12', vocab)
#     # bert_embedding = extract_embeddings('uncased_L-12_H-768_A-12', ["[PAD]"])
#     #
#     maxlen = max([len(sentence) for sentence in bert_embedding])
#     print(maxlen)
#     for i, sentence in enumerate(bert_embedding):
#         while len(bert_embedding[i]) < maxlen:
#             bert_embedding[i] = np.append(bert_embedding[i], [np.zeros(768)], axis=0)
#
#     with open('bert_embedding_sent.pkl', 'wb') as f:
#         pickle.dump(bert_embedding, f)
#         print('loaded and saved as a pickle')
#     # pad_emd = extract_embeddings('uncased_L-12_H-768_A-12', [" [PAD] "])
#     # print(len(pad_emd))
#
# maxlen = max([len(sentence) for sentence in bert_embedding])
# print(maxlen)
# # for i,sentence in enumerate(bert_embedding):
# #     while len(bert_embedding[i]) < maxlen:
# #         bert_embedding[i] = np.append(bert_embedding[i], [np.zeros(768)], axis=0)
# #     # print(len(bert_embedding[i]))
#
# X_train = bert_embedding[:len(train_sentences)]
# X_test = bert_embedding[len(train_sentences):]
#
#
#
# # print(X_train[:5])
# X_train = np.stack(X_train, axis=0)
# X_test = np.stack(X_test, axis=0)
# # print(X_train.shape)
#
# print(X_train.shape)




# input = Input(shape=(maxlen,768, ))

# if mode == 'BERT':
#     pass
#     # embedding = Embedding(input_dim=len(token_dict), output_dim=w2v_size, mask_zero=True, weights=[pretrained_weights], trainable=False)(input)
#
# elif mode == 'attention':
# embedding = Embedding(input_dim=vocab_size, output_dim=w2v_size, mask_zero=True, weights=[pretrained_weights],
#                           trainable=False)(input)

# bilstm = Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.3, return_state=False))(input)
# context_vector, attn = AttentionWeightedAverage(return_attention=True)(bilstm)
# dense = BatchNormalization()(context_vector)
# masking = Masking(mask_value=0.0)(input)
# dense = Dense(units=768, activation='tanh')(input)
# dense = BatchNormalization()(dense)
# dense = Dense(units=512, activation='tanh')(dense)
# dense = BatchNormalization()(dense)
# dense = Dense(units=256, activation='tanh')(dense)
# dense = Permute((2,1))(dense)
# dense = Dense(units=64, activation='tanh')(dense)
# dense = BatchNormalization()(dense)
# dense = Dense(units=32, activation='tanh')(dense)
# dense = BatchNormalization()(dense)
# dense = Permute((2,1))(dense)
# dense = Dense(units=64, activation='tanh')(dense)
# dense = BatchNormalization()(dense)
# dense = Dense(units=16, activation='tanh')(dense)
# dense = Flatten()(dense)
# dense = BatchNormalization()(dense)
# dense = Dense(units=16, activation='tanh')(dense)
# dense = BatchNormalization()(dense)
# output = Dense(units=1, activation='tanh')(dense)

# model = Model(inputs=input, outputs=output)
# model.add(Masking(mask_value=0.))

# adam = Adam(lr=0.0001)
# model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])
model.summary()

def call_corr(epoch, logs):

    train_predict = model.predict(x=X_train).flatten()
    test_predict = model.predict(x=X_test).flatten()

    # if not epoch % 10:
    train_corr = np.corrcoef(train_predict, train_label)
    test_corr = np.corrcoef(test_predict, test_label)
    print(f'train correlation : {train_corr[0][1]} , test correlation : {test_corr[0][1]}')

    train_bi_predict = train_predict > 0
    test_bi_predict = test_predict > 0

    train_bi_acc = (train_bi_predict == (np.array(train_label)>0)).mean()
    test_bi_acc = (test_bi_predict == (np.array(test_label)>0)).mean()

    print(f'train acc: {train_bi_acc} - test acc: {test_bi_acc}')
    print()
    if epoch > 10 and not epoch % 5:
        # print('-----------train high examples---------')
        # for i,v in enumerate(heapq.nlargest(5, range(len(train_predict)), train_predict.take)):
        #     print(" ".join(train_sentences[i]))
        # print(train_sentences[heapq.nlargest(5, range(len(train_predict)), train_predict.take)])
        print('-----------test high examaples---------')
        for i in heapq.nlargest(5, range(len(test_predict)), test_predict.take):
            print(" ".join(test_sentences[i]))
        print('-----------test low examples----------')
        for i in heapq.nsmallest(5, range(len(test_predict)), test_predict.take):
            print(" ".join(test_sentences[i]))

hist = model.fit(X_train, train_label, validation_data=[X_test, test_label], epochs=100, batch_size=BATCH_SIZE, verbose=2, callbacks=[LambdaCallback(on_epoch_end=call_corr)])

