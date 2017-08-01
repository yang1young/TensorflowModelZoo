# -*- coding:utf-8 -*-
import tensorflow as tf
import pandas as pd

class MLP():
    def __init__(self,x_dimention,y_class,hidden_num):
        self.x = tf.placeholder(tf.float32, [None, x_dimention])
        self.y = tf.placeholder(tf.float32, [None, y_class])
        self.keep_prob = tf.placeholder(tf.float32)

        def get_layer(X, input_dimention, output_dimention, active_function):
            # 在tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择。这样保证了生成的值都在均值附近。
            weight = tf.Variable(tf.truncated_normal([input_dimention, output_dimention], mean=0.0, stddev=0.1))
            bias = tf.Variable(tf.truncated_normal([output_dimention], mean=0.0, stddev=0.1))
            return active_function(tf.add(tf.matmul(X, weight), bias))

        layer1 = tf.nn.dropout(get_layer(self.x, x_dimention, hidden_num, tf.nn.tanh), self.keep_prob)
        layer2 = tf.nn.dropout(get_layer(layer1, hidden_num, hidden_num, tf.nn.tanh), self.keep_prob)
        layer3 = tf.nn.dropout(get_layer(layer2, hidden_num, hidden_num, tf.nn.tanh), self.keep_prob)
        tf.summary.histogram('linear', layer3)
        with tf.name_scope('output'):
            self.logits = get_layer(layer3, hidden_num, y_class, tf.nn.tanh)
            self.predictions = tf.argmax(self.logits, 1, name='predictions')

        with tf.name_scope('loss'):
            # tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)
            # logits：神经网络最后一层的输出，大小是[batchsize，num_classes],logits是未经缩放的, labels：实际标签
            # 第一步先对网络最后一层的输出做softmax，输出是num_classes大小的向量（[Y1，Y2,Y3...]其中Y1，Y2，Y3...分别代表了是属于该类的概率）
            # 第二步是softmax的输出向量[Y1，Y2,Y3...]和样本的实际标签做一个交叉熵
            # tf.nn.softmax_cross_entropy_with_logits可以由tf.nn.softmax+cross_entropy replace
            # 如果labels的每一行是one-hot表示，也就是只有一个地方为1，其他地方为0，可以使用tf.sparse_softmax_cross_entropy_with_logits()
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = self.logits))

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.name_scope('num_correct'):
            correct = tf.equal(self.predictions, tf.argmax(self.y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct, 'float'))

        #add summary
        tf.summary.scalar("loss",self.loss)
        #add summary
        tf.summary.scalar("accuracy_summary",self.accuracy)
        self.summary =  tf.summary.merge_all()
        #tf.summary.merge([loss_summary,accuracy_summary])


'''
（1）SCALARS
展示的是标量的信息，我程序中用tf.summary.scalars()定义的信息都会在这个窗口。 
（5）DISTRIBUTIONS
这里查看的是神经元输出的分布，有激活函数之前的分布，激活函数之后的分布等。 
（6）HISTOGRAMS
也可以看以上数据的直方图 
'''