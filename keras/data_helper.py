from __future__ import print_function
import os
import sys
import random
import numpy as np
import pandas as pd
from clean_utils import clean_utils
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

reload(sys)
sys.setdefaultencoding('utf-8')
K.set_image_dim_ordering('th')

# get pre-trained word embedding
def get_embedding(vetctor_name):
    print('Indexing word vectors.')
    embeddings_index = {}
    f = open(vetctor_name)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


# vectorize the text samples into a 2D integer tensor
def token_padding(texts, max_word_token, max_seq_length):
    tokenizer = Tokenizer(nb_words=max_word_token, split=' ')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=max_seq_length)
    print('Shape of data tensor:', data.shape)
    return data, word_index


# process labels
def label_to_categorical(labels, need_to_categorical):
    label = sorted(list(set(labels)))
    num_labels = len(label)
    print('label total count is: ' + str(num_labels))
    label_indict = range(num_labels)
    labels_index = dict(zip(label, label_indict))
    labels = [labels_index[y] for y in labels]
    if (need_to_categorical):
        labels = to_categorical(np.asarray(labels))
    print('Shape of label tensor:', labels.shape)
    return labels, label


# split train_test_dev data
def train_test_split(data, labels, split_percent, need_dev_data):
    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(split_percent * data.shape[0])
    if (need_dev_data):
        train_end = -nb_validation_samples * 3
        val_end = -nb_validation_samples * 2
    else:
        train_end = -nb_validation_samples * 2
        val_end = train_end

    x_train = data[:train_end]
    y_train = labels[:train_end]
    x_test = data[val_end:]
    y_test = labels[val_end:]
    if (need_dev_data):
        x_val = data[train_end:val_end]
        y_val = labels[train_end:val_end]
    else:
        x_val = None
        y_val = None
    return x_train, y_train, x_val, y_val, x_test, y_test


# load raw data
def data_load(is_csv, data_dir, text_name, labels_name):
    # second, prepare text samples and their labels
    print('Processing text dataset')
    if (is_csv):
        df = pd.read_csv(data_dir + text_name, sep='@', header=None, encoding='utf8', engine='python')
        selected = ['Code', 'Tag']
        df.columns = selected
        texts = (df[selected[0]]).tolist()
        texts = [s.encode('utf-8') for s in texts]
        labels = (df[selected[1]]).tolist()
    else:
        texts = open(data_dir + text_name, 'r').readlines()
        texts = [s.encode('utf-8').replace('\n', '') for s in texts]
        labels = open(data_dir + labels_name, 'r').readlines()
        labels = [s.encode('utf-8').replace('\n', '') for s in labels]
    print('Found %s texts.' % len(texts))
    return texts, labels


# evaluate trained classification models
def eval_model(predict, groud_truth, label, log_path):
    count_set = []
    predict_set = []
    right_set = []
    precision_list = []
    recall_list = []
    for i in range(len(label)):
        count_set.append(0.0)
        predict_set.append(0.0)
        right_set.append(0.0)
        precision_list.append('0')
        recall_list.append('0')

    for p, r in zip(predict, groud_truth):
        indexP = np.argmax(p)
        indexR = np.argmax(r)
        count_set[int(indexR)] += 1
        predict_set[int(indexP)] += 1
        if (int(indexR) == int(indexP)):
            right_set[int(indexP)] += 1
    count = 0
    result_file = open(log_path + 'Result.csv', 'w')

    for i, j, k in zip(count_set, predict_set, right_set):
        tag = str(label[count])
        if (i == 0):
            recall = '0'
        else:
            recall = str(k / i)
        if (j == 0):
            precision = '0'
        else:
            precision = str(k / j)
        precision_list.append(precision)
        recall_list.append(recall)
        print(tag + ',' + str(i) + ',' + str(precision) + ',' + str(recall))
        result_file.write(tag + ',' + str(i) + ',' + str(precision) + ',' + str(recall) + '\n')
        count += 1
    return precision_list, recall_list, label


if __name__ == "__main__":
    data_load(False, '/home/yangqiao/pythonProject/DeeplearningModels/data/', 'train', 'labels')
