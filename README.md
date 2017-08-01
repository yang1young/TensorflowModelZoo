# TensorflowModelZoo
Some deeplearning models for NLP,image,ML based on Keras and Tensorflow
* CNN LSTM to solve text classification
* Tf-idf and N-gram for text classification
* Pre-trained word embedding using Glove
* Tools to validate and plot using Prettytable and Matplotlib
* Multi-layer perception model
* Dual input text classification
* Implements of famous deepleanrning models using tensorflow

## Environment required
   All models are tested based on Tensorflow 1.0.0, other verison may have some API change

## File Introduction
1. clean_utils
   - Text clean utils
2. CNN  

   Some models of CNN,
   - Mnist_Softmax.py  
   Using softmax to solve Mnist handwritten digits classification
   The MNIST database of handwritten digits, has a training set of 60,000 examples,
   and a test set of 10,000 examples. It is a subset of a larger set available from NIST.
   The digits have been size-normalized and centered in a fixed-size image.
   http://yann.lecun.com/exdb/mnist/
   - Mnist_MLP.py  
   Using MLP to solve Mnist handwritten digits classification
   Using dropout and one hidden layer
   - Mnist_LeNet.py  
   Using LeNet5 to solve Mnist handwritten digits classification
   LeNet is a Convolutional Neural Network, using maxpooling and conv2d
   http://yann.lecun.com/exdb/lenet/
   - Mnist_AlexNet.py   
   [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    AlexNet implements
   - text_classification_cnn   
    http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
3. keras  
   Models implements using keras
   - data_helper.py  
   data prepare
   - dual_input_cnn_rnn.py
   dual input cnn and rnn model to do text classification
   - keras_basic.py
   basic practice using keras
   - rf_tfidf_text_classification.py
   sklearn tf-idf text classification
   - text_classification_cnn_rnn.py
   cnn and rnn to do text classification

4. MLP  
   Multi-layer perception model
   - data_helper.py  
     prepare data
   - Model.py  
     neural network model
   - predict.py  
     reload model and prediction
   - train.py  
     train model with fresh parameters
5. notebook  
   Jupyter-notebook tensorflow practice code
6. RNN  
   Models of RNN
   - text_classification_cnn_rnn
    from https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn
   - text_classification_rnn
   https://github.com/luchi007/RNN_Text_Classify
   https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn
7. translate  
   Google tensorflow seq2seq model, from https://github.com/tensorflow/models
8. fp_growth.py  
A pure Python implementation of the FP-growth algorithm for finding frequent itemsets.
from :https://github.com/enaeseth/python-fp-growth.git

## Tips
### Something about keras and tensorflow
1. http://wiki.jikexueyuan.com/project/tensorflow-zh/
2. https://keras.io/

### About Glove
Similar to word2vec,GloVe is an unsupervised learning algorithm for obtaining vector representations
for words. Training is performed on aggregated global word-word co-occurrence statistics
from a corpus, and the resulting representations showcase interesting linear substructures of
the word vector space.
1. https://nlp.stanford.edu/projects/glove/
2. https://flystarhe.github.io/2016/09/04/word2vec-test/

### If you think this repo may help you, may you star on my project :)