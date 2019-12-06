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
from keras_bert import extract_embeddings, Tokenizer

K.tensorflow_backend._get_available_gpus()


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

# w2v_size = 300

if which_embedding is None:

    print('embedding pretrain...')
    if False:
        w2v_model = FastText([['<dummy>']] + train_sentences + test_sentences, sg=1, min_count=1, sorted_vocab=False, size=w2v_size)
        with open('embedding_pretrain.pkl', 'wb') as f:
            pickle.dump(w2v_model, f)
    else:
        with open('embedding_pretrain.pkl', 'rb') as f:
            w2v_model = pickle.load(f)
elif which_embedding == 'Google_W2V':
    if False:
        print('creating google news word2vec...')
        FILENAME = "GoogleNews-vectors-negative300.bin.gz"
        w2v_model = KeyedVectors.load_word2vec_format(FILENAME, binary=True, limit=3000000)
        corpus_vocab = set(word for sentence in train_sentences+test_sentences for word in sentence)
        diff = list(corpus_vocab.difference(w2v_model.vocab))
        print(len(corpus_vocab))
        print(f'lacking {len(diff)} words')
        a = np.var(w2v_model.vectors)
        print('dumping w2v...')
        w2v_model.add(entities=diff, weights=np.random.uniform(-a, -a, (len(diff), w2v_size)))
        del corpus_vocab
        del a

        with open('w2v_google.pkl', 'wb') as f:
            pickle.dump(w2v_model, f, protocol=4)

    else:
        print('loading saved google news word2vec...')
        with open('w2v_google.pkl', 'rb') as f:
            w2v_model = pickle.load(f)




    print(w2v_model.wv.index2word[0])
    print(w2v_model.wv.vocab['</s>'].index) #dummy index 0인지 확인



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

try:
    with open('bert_embedding_sent.pkl', 'rb') as f:
        print('loading existing bert embedding...')
        bert_embedding = pickle.load(f)
        print('loaded')
except:
    print('loading bert embedding')
    bert_embedding = extract_embeddings('uncased_L-12_H-768_A-12', [" ".join(sentence) for sentence in train_sentences + test_sentences])
    # bert_embedding = extract_embeddings('uncased_L-12_H-768_A-12', vocab)
    # bert_embedding = extract_embeddings('uncased_L-12_H-768_A-12', ["[PAD]"])
    #
    maxlen = max([len(sentence) for sentence in bert_embedding])
    print(maxlen)
    for i, sentence in enumerate(bert_embedding):
        while len(bert_embedding[i]) < maxlen:
            bert_embedding[i] = np.append(bert_embedding[i], [np.zeros(768)], axis=0)

    with open('bert_embedding_sent.pkl', 'wb') as f:
        pickle.dump(bert_embedding, f)
        print('loaded and saved as a pickle')
    # pad_emd = extract_embeddings('uncased_L-12_H-768_A-12', [" [PAD] "])
    # print(len(pad_emd))

maxlen = max([len(sentence) for sentence in bert_embedding])
print(maxlen)
# for i,sentence in enumerate(bert_embedding):
#     while len(bert_embedding[i]) < maxlen:
#         bert_embedding[i] = np.append(bert_embedding[i], [np.zeros(768)], axis=0)
#     # print(len(bert_embedding[i]))

X_train = bert_embedding[:len(train_sentences)]
X_test = bert_embedding[len(train_sentences):]



# print(X_train[:5])
X_train = np.stack(X_train, axis=0)
X_test = np.stack(X_test, axis=0)
# print(X_train.shape)

print(X_train.shape)




input = Input(shape=(maxlen,768, ))

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
dense = Dense(units=768, activation='tanh')(input)
dense = BatchNormalization()(dense)
dense = Dense(units=512, activation='tanh')(dense)
dense = BatchNormalization()(dense)
dense = Dense(units=256, activation='tanh')(dense)
dense = Permute((2,1))(dense)
dense = Dense(units=64, activation='tanh')(dense)
dense = BatchNormalization()(dense)
dense = Dense(units=32, activation='tanh')(dense)
dense = BatchNormalization()(dense)
dense = Permute((2,1))(dense)
dense = Dense(units=64, activation='tanh')(dense)
dense = BatchNormalization()(dense)
dense = Dense(units=16, activation='tanh')(dense)
dense = Flatten()(dense)
dense = BatchNormalization()(dense)
dense = Dense(units=16, activation='tanh')(dense)
dense = BatchNormalization()(dense)
output = Dense(units=1, activation='tanh')(dense)

model = Model(inputs=input, outputs=output)
# model.add(Masking(mask_value=0.))

adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])
model.summary()

train_corr_list = []
test_corr_list = []

def call_corr(epoch, logs):

    train_predict = model.predict(x=X_train).flatten()
    test_predict = model.predict(x=X_test).flatten()

    # if not epoch % 10:
    train_corr = np.corrcoef(train_predict, train_label)
    test_corr = np.corrcoef(test_predict, test_label)
    print(f'train correlation : {train_corr[0][1]} , test correlation : {test_corr[0][1]}')
    train_corr_list.append(train_corr[0][1])
    test_corr_list.append(test_corr[0][1])

    # train_bi_predict = train_predict > 0
    # test_bi_predict = test_predict > 0

    # train_bi_acc = (train_bi_predict == (np.array(train_label)>0)).mean()
    # test_bi_acc = (test_bi_predict == (np.array(test_label)>0)).mean()
    #
    # print(f'train acc: {train_bi_acc} - test acc: {test_bi_acc}')
    # print()
    # if epoch > 10 and not epoch % 5:
    #     # print('-----------train high examples---------')
    #     # for i,v in enumerate(heapq.nlargest(5, range(len(train_predict)), train_predict.take)):
    #     #     print(" ".join(train_sentences[i]))
    #     # print(train_sentences[heapq.nlargest(5, range(len(train_predict)), train_predict.take)])
    #     print('-----------test high examaples---------')
    #     for i in heapq.nlargest(5, range(len(test_predict)), test_predict.take):
    #         print(" ".join(test_sentences[i]))
    #     print('-----------test low examples----------')
    #     for i in heapq.nsmallest(5, range(len(test_predict)), test_predict.take):
    #         print(" ".join(test_sentences[i]))

history = model.fit(X_train, train_label, validation_data=[X_test, test_label], epochs=100, batch_size=128, verbose=2, callbacks=[LambdaCallback(on_epoch_end=call_corr)])

history.history['train_corr'] = train_corr_list
history.history['test_corr'] = test_corr_list

with open('bert_feature.pkl', 'wb') as f:
    pickle.dump(history.history, f)
