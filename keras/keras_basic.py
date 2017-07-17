#!/usr/bin/python
# coding=utf-8

#layer
'''
layer.get_weights()：返回层的权重（numpy array）
layer.set_weights(weights)：从numpy array中将权重加载到该层中，要求与* layer.get_weights()的形状相同
layer.get_config()：返回当前层配置信息的字典，层也可以借由配置信息重构

如果层仅有一个计算节点（即该层不是共享层），则可以通过下列方法获得输入张量、输出张量、输入数据的形状和输出数据的形状：
layer.input
layer.output
layer.input_shape
layer.output_shape
'''

#
'''
keras.layers.core.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
Dense就是常用的全连接层，所实现的运算是output = activation(dot(input, kernel)+bias)。其中activation是逐元素计算的激活函数，kernel是本层的权值矩阵，bias为偏置向量，只有当use_bias=True才会添加。
如果本层的输入数据的维度大于2，则会先被压为与kernel相匹配的大小。

Permute层:keras.layers.core.Permute(dims)

Permute层将输入的维度按照给定模式进行重排，例如，当需要将RNN和CNN网络连接时，可能会用到该层。
参数dims：整数tuple，指定重排的模式，不包含样本数的维度。重拍模式的下标从1开始。例如（2，1）代表将输入的第二个维度重拍到输出的第一个维度，而将输入的第一个维度重排到第二个维度

例子
model = Sequential()
model.add(Permute((2, 1), input_shape=(10, 64)))
# now: model.output_shape == (None, 64, 10)
# note: `None` is the batch dimension

'''



