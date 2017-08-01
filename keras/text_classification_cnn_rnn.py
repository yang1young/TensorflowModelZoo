from __future__ import print_function

import os
import sys
import numpy as np
import data_helper as dl
from keras import backend as K
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding

reload(sys)
sys.setdefaultencoding('utf-8')
K.set_image_dim_ordering('th')

'''
using pre-trained word embeddings (GloVe embeddings) to do text classification
'''

# Path
DATADIR = '/home/qiaoyang/codeData/stackoverflow/'
TRAIN_DATA_NAME = 'train.csv'
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


# first, build index mapping words in the embeddings set
# to their embedding vector

def get_train_test_data(is_csv):
    embedding_index = dl.get_embedding(os.path.join(GLOVE_DIR, WORD_VECTOR_NAME))
    data, labels = dl.data_load(is_csv, DATADIR, TRAIN_DATA_NAME, LABELS_DATA_NAME)
    labels, label = dl.label_to_categorical(labels, True)
    data, word_index = dl.token_padding(data, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
    x_train, y_train, x_val, y_val, x_test, y_test = dl.train_test_split(data, labels, VALIDATION_SPLIT, True)

    return embedding_index, word_index, label, x_train, x_train, y_train, x_val, y_val, x_test, y_test


def train(is_csv):
    embedding_index, word_index, label, x_train, x_train, y_train, x_val, y_val, x_test, y_test = get_train_test_data(
        is_csv)
    print('Preparing embedding matrix.')

    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embedding_index.get(word)
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

    # train a 1D convnet with global maxpooling
    # sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # embedded_sequences = embedding_layer(sequence_input)
    # x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    # x = MaxPooling1D(5)(x)
    # x = Conv1D(128, 5, activation='relu')(x)
    # x = MaxPooling1D(5)(x)
    # x = Conv1D(128, 5, activation='relu')(x)
    # x = MaxPooling1D(3)(x)
    # x = Flatten()(x)
    # x = Dense(128, activation='relu')(x)
    # preds = Dense(len(labels_index), activation='softmax')(x)

    # model = Model(sequence_input, preds)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.25))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(CLASS_NUM))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    # happy learning!
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              nb_epoch=nb_epoch, batch_size=batch_size)
    predict = model.predict(x_test)
    dl.eval_model(predict, y_test, label, DATADIR)
    model_json = model.to_json()
    with open(MODEL_SAVE_DIR + "model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(MODEL_SAVE_DIR + "model.h5")
    print("Saved model to disk")
    # print(labels_index.get()i/j)


def reload_model():
    # load json and create model
    json_file = open(MODEL_SAVE_DIR + 'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy',
                         optimizer='rmsprop',
                         metrics=['acc'])


if __name__ == "__main__":
    train(True)
