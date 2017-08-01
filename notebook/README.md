# DeeplearningModels
Some deeplearning models to solve NLP problems based on Keras and Tensorflow
* CNN LSTM to solve Text classification
* pre-trained word embedding using Glove
* tools to validate and plot using Prettytable and Matplotlib
* some other models about deeplearning

## File introduce
1. text_classification_cnn_rnn
For keras_implements:
* dl_modules_classification.py some modules to prepare data
* text_classification_cnn_rnn.py  using cnn/LSTM to solve text classification problems
* dual_input_cnn_rnn.py  dual input cnn/LSTM to do text classification, you can input two kind of text once
* RF_tfidf_text_classification.py  TF-IDF and decision tree to solve text classification
* validate_tools.py  some tools to do result compare

For tensorflow_implements:
http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
http://blog.csdn.net/u010223750/article/details/53334313
https://github.com/luchi007/RNN_Text_Classify
https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn

2. translate
Google tensorflow seq2seq model, from https://github.com/tensorflow/models
3. fp_growth.py
a pure Python implementation of the FP-growth algorithm for finding frequent itemsets.
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

