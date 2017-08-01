from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import text_classification_cnn_rnn.keras_implements.data_helper as dl
from random import shuffle
from sklearn import tree
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

DATADIR = '/home/yangqiao/pythonProject/DeeplearningModels/data/'
MAX_SEQUENCE_LENGTH = 200
MAX_NB_WORDS = 5000
EMBEDDING_DIM = 400
VALIDATION_SPLIT = 0.1
TRAIN_DATA_NAME = 'train'
LABELS_DATA_NAME = 'labels'

#shuffle data
def shuffles(list1,list2):
    list1_shuf = []
    list2_shuf = []
    index_shuf = range(len(list1))
    shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(list1[i])
        list2_shuf.append(list2[i])
    return list1_shuf,list2_shuf


#prepare training and testing data
#using TF-idf
def data_prepare(is_csv):
    data, labels = dl.data_load(is_csv, DATADIR, TRAIN_DATA_NAME, LABELS_DATA_NAME)
    labels, label = dl.label_to_categorical(labels, True)
    data,labels = shuffles(data,labels)

    nb_validation_samples = int(VALIDATION_SPLIT * (len(labels)))
    x_train = data[:-nb_validation_samples * 3]
    y_train = labels[:-nb_validation_samples * 3]
    x_test = data[-nb_validation_samples*3:]
    y_test = labels[-nb_validation_samples*3:]

    count_vect = CountVectorizer(ngram_range=(1,1),lowercase=False, binary=True,min_df=1,max_features=25000)
    X_train_counts = count_vect.fit_transform(x_train)
    print X_train_counts.toarray()
    print count_vect.get_feature_names()
    tfidf_transformer = TfidfTransformer()
    x_train = tfidf_transformer.fit_transform(X_train_counts)
    print x_train.toarray()

    x_test = count_vect.transform(x_test)
    x_test = tfidf_transformer.transform(x_test)
    return x_train,y_train,x_test,y_test,label


def train(is_csv):
    x_train, y_train, x_test, y_test, label = data_prepare(is_csv)
    #clf = MultinomialNB().fit(x_train, y_train)
    clf = tree.DecisionTreeClassifier().fit(x_train, y_train)
    predict = clf.predict(x_test)
    dl.eval_model(predict,y_test,label,DATADIR)


if __name__ == "__main__":
    train(False)
