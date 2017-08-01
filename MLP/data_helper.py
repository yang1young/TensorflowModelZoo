import pandas as pd
import numpy as np
import json

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def load_data(path,filename):
    df = pd.read_csv(path+filename,header=None)
    data = df.reindex(np.random.permutation(df.index))
    x_raw= data.iloc[:,:len(data.columns.tolist())-1].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

    labels = sorted(list(set(data.iloc[:,-1].tolist())))
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    with open(path + 'labels.json', 'w') as outfile:
        json.dump(labels, outfile, indent=4, ensure_ascii=False)
    y_raw = data.iloc[:,-1].apply(lambda y: label_dict[y]).tolist()
    x = np.array(x_raw)
    y = np.array(y_raw)
    return x,y

def load_unseen_data(path,filename):
    labels = json.loads(open(path + 'labels.json').read())
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    df = pd.read_csv(path+filename,header=None)
    data = df.reindex(np.random.permutation(df.index))
    x_raw= data.iloc[:,:len(data.columns.tolist())-1].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    y_raw = data.iloc[:,-1].apply(lambda y: label_dict[y]).tolist()
    x = np.array(x_raw)
    y = np.array(y_raw)
    return x,y

if __name__ == "__main__":
    datapath = '/home/yang/PythonProject/DeeplearningModels/MLP/data.csv'
    load_data(datapath,'')
