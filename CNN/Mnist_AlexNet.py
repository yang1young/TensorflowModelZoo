# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tflearn.datasets.oxflower17 as oxflower17
import numpy as np

#X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))
mnist = input_data.read_data_sets("data/mnist/",one_hot=True)

# Parameters
learning_rate = 0.001
batch_size = 32

# Network Parameters
n_input = 28*28 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.8 # Dropout, probability to keep units


#print tensor shape for every layer
def print_Activations(t):
    print (t.op.name,' ',t.get_shape().as_list())

def build_graph(images):
    #first layer conv,11x11x64 kernel,4x4 strides
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11,11,1,64],dtype=tf.float32,stddev=1e-1),name ='weights')
        conv = tf.nn.conv2d(images,kernel,[1,4,4,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),trainable=True,name = 'biases')
        bias = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(bias,name=scope)
        print_Activations(conv1)

    #3x3 pooling, strides 2x2, no zero padding
    lrn1 = tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9,beta=0.75,name = 'lrn1')
    pool1 = tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name = 'pool1')
    print_Activations(pool1)

    #second layer conv,5x5x192 kernel
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,stddev=1e-1),name = 'weight')
        conv = tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[192],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(bias,name = scope)
    print_Activations(conv2)

    #3x3 pooling, strides 2x2, no zero padding
    lrn2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9,beta=0.75,name = 'lrn2')
    pool2 = tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name = 'pool2')
    print_Activations(pool2)

    #third layer conv,3x3x384,strides = 1
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,192,384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        print_Activations(conv3)

    # fourth layer conv,3x3x256,strides = 1
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        print_Activations(conv4)

    # fifth layer conv,3x3x256,strides = 1
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        print_Activations(conv5)

    # 3x3 pooling, strides 2x2, no zero padding
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
    print_Activations(pool5)

    with tf.name_scope('fc6') as scope:
        weights = tf.Variable(tf.truncated_normal([9216, 4096], dtype=tf.float32, stddev=1e-1), name='fc6W')
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='fc6b')
        fc6 = tf.nn.relu_layer(tf.reshape(pool5, [-1, 9216]), weights, biases)

    ### Seventh layer. Fully connected ###
    with tf.name_scope('fc7') as scope:
        weights = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1), name='fc7W')
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='fc7b')
        fc7 = tf.nn.relu_layer(fc6, weights, biases)

    ### Eighth layer ###
    with tf.name_scope('fc8') as scope:
        weights = tf.Variable(tf.truncated_normal([4096,n_classes], dtype=tf.float32, stddev=1e-1), name='fc8W')
        biases = tf.Variable(tf.constant(0.0, shape=[n_classes], dtype=tf.float32), trainable=True, name='fc8b')
        fc8 = tf.nn.xw_plus_b(fc7, weights, biases)

    ### Probability. SoftMax ###
    y = tf.nn.softmax(fc8)
    return y


if __name__ == "__main__":
    session = tf.InteractiveSession()
    # tf Graph input
    x = tf.placeholder(tf.float32, [32, n_input])
    x_image = tf.reshape(x,[-1,28,28,1])
    y_ = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

    y = build_graph(x_image)

    # define loss
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)

    # training
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.global_variables_initializer().run()
    for i in range(20000):
        batch = mnist.train.next_batch(batch_size)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d,training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # evaluate
    print (accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


'''

第一层卷积层 输入图像为227*227*3(paper上貌似有点问题224*224*3)的图像，使用了96个kernels（96,11,11,3），
            以4个pixel为一个单位来右移或者下移，能够产生5555个卷积后的矩形框值，
            然后进行response-normalized（其实是Local Response Normalized，后面我会讲下这里）
            和pooled之后，pool这一层好像caffe里面的alexnet和paper里面不太一样，
            alexnet里面采样了两个GPU，所以从图上面看第一层卷积层厚度有两部分，池化pool_size=(3,3),
            滑动步长为2个pixels，得到96个2727个feature。
第二层卷积层使用256个（同样，分布在两个GPU上，每个128kernels（5*5*48）），做pad_size(2,2)的处理，
            以1个pixel为单位移动（感谢网友指出），能够产生27*27个卷积后的矩阵框，做LRN处理，然后pooled，
            池化以3*3矩形框，2个pixel为步长，得到256个13*13个features。
第三层第四层都没有LRN和pool，第五层只有pool，其中第三层使用384个kernels（3*3*384，pad_size=(1,1),
        得到384*15*15，kernel_size为（3，3),以1个pixel为步长，得到384*13*13）；
        第四层使用384个kernels（pad_size(1,1)得到384*15*15，核大小为（3，3）步长为1个pixel，
        得到384*13*13）；第五层使用256个kernels（pad_size(1,1)得到384*15*15，
        kernel_size(3,3)，得到256*13*13，pool_size(3，3）步长2个pixels，得到256*6*6）。
全连接层： 前两层分别有4096个神经元，最后输出softmax为1000个（ImageNet），
        注意caffe图中全连接层中有relu、dropout、innerProduct。

'''