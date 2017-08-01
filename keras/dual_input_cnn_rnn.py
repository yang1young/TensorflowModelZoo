from __future__ import print_function
import os
import numpy as np
import pandas as pd
from keras.layers import Merge
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import LSTM
import data_helper as dl
from keras.layers import Convolution1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras import backend as K

K.set_image_dim_ordering('th')
from keras.models import model_from_json

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

# Path
DATADIR = '/home/qiaoyang/codeData/stackoverflow/'
TRAIN_DATA_NAME1 = 'train.csv'
TRAIN_DATA_NAME2 = 'train.csv'
LABELS_DATA_NAME = 'tags'
GLOVE_DIR = '/home/qiaoyang/codeData/code/codeRemoveData/glove/GloVe-1.2'
WORD_VECTOR_NAME = 'code.vectors.glove.txt'
MODEL_SAVE_DIR = ''

# Data
MAX_SEQUENCE_LENGTH = 200
MAX_NB_WORDS = 5000
EMBEDDING_DIM = 400
VALIDATION_SPLIT = 0.1
CLASS_NUM = 84

# Convolution
filter_length = 5
nb_filter = 64
pool_length = 4

# LSTM
lstm_output_size = 150

# training
nb_epoch = 10
batch_size = 128


def getTrain(file_name, need_label, is_csv):
    if (is_csv):
        df = pd.read_csv(DATADIR + file_name, sep='@', header=None, encoding='utf8', engine='python')
        selected = ['Code', 'Tag']
        df.columns = selected
        texts = (df[selected[0]]).tolist()
        texts = [s.encode('utf-8') for s in texts]
        if (need_label):
            label = sorted(list(set(df[selected[1]].tolist())))
            num_labels = len(label)
            print(label)
            lableIndict = range(num_labels)
            labels_index = dict(zip(label, lableIndict))
            labels = df[selected[1]].apply(lambda y: labels_index[y]).tolist()
            return texts, labels, label
        else:
            return texts
    else:
        df = pd.read_csv(DATADIR + 'tags.csv', sep='@', header=None, encoding='utf8', engine='python')
        selected = ['Tag']
        df.columns = selected
        texts = open(DATADIR + file_name, 'r')
        texts = [s.encode('utf-8') for s in texts]

        if (need_label):
            label = sorted(list(set(df[selected[0]].tolist())))
            num_labels = len(label)
            lableIndict = range(num_labels)
            labels_index = dict(zip(label, lableIndict))
            labels = df[selected[0]].apply(lambda y: labels_index[y]).tolist()
            return texts, labels, label
        else:
            return texts


def get_token(texts):
    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, split=' ')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return data, word_index


def dataPrepare(is_csv):
    print('Indexing word vectors.')
    embedding_index = dl.get_embedding(os.path.join(GLOVE_DIR, WORD_VECTOR_NAME))

    data1, labels = dl.data_load(is_csv, DATADIR, TRAIN_DATA_NAME1, LABELS_DATA_NAME)
    data2, _ = dl.data_load(is_csv, DATADIR, TRAIN_DATA_NAME2, LABELS_DATA_NAME)
    source_data_token, source_word_index = dl.token_padding(data1, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
    type_data_token, type_word_index = dl.token_padding(data2, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
    labels, label = dl.label_to_categorical(labels, True)

    # split the data into a training set and a validation set
    indices = np.arange(source_data_token.shape[0])
    np.random.shuffle(indices)
    source_data_token = source_data_token[indices]
    type_data_token = type_data_token[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * source_data_token.shape[0]) * 2

    source_train = source_data_token[:-nb_validation_samples]
    source_test = source_data_token[-nb_validation_samples:]
    type_train = type_data_token[:-nb_validation_samples]
    type_test = type_data_token[-nb_validation_samples:]
    label_train = labels[:-nb_validation_samples]
    label_test = labels[-nb_validation_samples:]

    return source_train, source_test, type_train, type_test, \
           label_train, label_test, source_word_index, type_word_index, embedding_index, label


def get_model(word_index, embeddings_index):
    print('Preparing embedding matrix.')
    # num_words = min(MAX_NB_WORDS, len(word_index))
    num_words = 20000
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    print('Training model.')
    model1 = Sequential()
    model1.add(embedding_layer)
    model1.add(Dropout(0.25))
    model1.add(Convolution1D(nb_filter=nb_filter,
                             filter_length=filter_length,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    model1.add(MaxPooling1D(pool_length=pool_length))
    model1.add(LSTM(lstm_output_size))
    model1.add(Dense(CLASS_NUM))
    return model1


def train(is_csv):
    source_train, source_test, type_train, type_test, \
    label_train, label_test, source_word_index, type_word_index, embeddings_index, label = dataPrepare(is_csv)

    model1 = get_model(source_word_index, embeddings_index)
    print(source_word_index)
    model3 = get_model(type_word_index, embeddings_index)

    merged = Merge([model1, model3], mode='concat')

    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(CLASS_NUM, activation='softmax'))

    final_model.compile(loss='categorical_crossentropy',
                        optimizer='rmsprop',
                        metrics=['acc'])

    final_model.fit([source_train, type_train], label_train, nb_epoch=nb_epoch, batch_size=batch_size)
    predict = final_model.predict([source_test, type_test])
    dl.eval_model(predict, label_test, label, DATADIR)


if __name__ == "__main__":
    train(False)
