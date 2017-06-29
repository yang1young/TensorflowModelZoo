import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/mnist/",one_hot=True)

session = tf.InteractiveSession()

in_unit = 784
h1_unit = 300
W1 = tf.Variable(tf.truncated_normal([in_unit,h1_unit],stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_unit]))
W2 = tf.Variable(tf.zeros([h1_unit,10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32,[None,in_unit])
# parameter for dropout
keep_prob = tf.placeholder(tf.float32)
# hidden layer
hidden1 = tf.nn.relu(tf.matmul(x,W1)+b1)
hidden1_drop = tf.nn.dropout(hidden1,keep_prob)

y = tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)
y_ = tf.placeholder(tf.float32,[None,10])
#define loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

tf.global_variables_initializer().run()

for i in range(5000):
    batch_x,batch_y = mnist.train.next_batch(100)
    train_step.run({x:batch_x,y_:batch_y,keep_prob:0.75})
#evaluate
correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print (accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))