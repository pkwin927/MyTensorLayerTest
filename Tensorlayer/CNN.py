#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 04:08:02 2018

@author: makise
"""


import tensorflow as tf
import tensorlayer as tl
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()
tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001     

mnist = input_data.read_data_sets('./mnist', one_hot = True)

test_x = mnist.test.images[:20]

test_y = mnist.test.labels[:20]
#X_train, y_train, X_val, y_val, X_test, y_test = tl.filter.load_mnist_dataset(shape = (-1,28,28,1))

#batch_size = 128

#x = tf.placeholder(tf.float32,)

tf_x = tf.placeholder(tf.float32, [None, 28*28]) /255.

image = tf.reshape(tf_x, [-1,28,28,1])

tf_y = tf.placeholder(tf.int32, [None, 10])

network = tl.layers.InputLayer(image, name = 'input')

network = tl.layers.Conv2d(network, n_filter = 16, filter_size = (5,5), strides = (1,1), act = tf.nn.relu, padding = "SAME", name = 'cnn1')

network = tl.layers.MaxPool2d(network, filter_size = (2,2), strides = (2,2), padding = 'SAME', name = 'pool1')

network = tl.layers.Conv2d(network, n_filter = 32, filter_size = (5,5), strides = (1,1), act = tf.nn.relu, padding = 'SAME', name = 'cnn2')

network = tl.layers.MaxPool2d(network,  filter_size = (2,2), strides = (2,2), padding = "SAME", name = 'pool2')

network = tl.layers.FlattenLayer(network, name = 'flatten')

#network = tl.layers.DropoutLayer(network, keep = 0.5, name = 'drop1'

#network = tl.layers.DenseLayer(network, 256, tf.nn.relu, name = 'relu1')

#network = tl.layers.DropoutLayer(network, keep = 0.5, name = 'drop2')

network = tl.layers.DenseLayer(network, 10, tf.identity, name = 'output')

output = network.outputs

loss = tf.losses.softmax_cross_entropy(onehot_labels = tf_y, logits = output)

train_op = tf.train.AdamOptimizer(learning_rate = LR).minimize(loss)

accuracy = tf.metrics.accuracy(labels = tf.argmax(tf_y,axis = 1), predictions = tf.argmax(output, axis = 1),)[1]

sess = tf.Session()

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

sess.run(init_op)

for step in range(600):
    
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    
    _, loss_ = sess.run([train_op, loss], {tf_x:b_x, tf_y:b_y})
    
    if step % 50 == 0:
        
        accuracy_ = sess.run(accuracy, {tf_x: test_x, tf_y: test_y})
#        print('test accuracy: %.2f' , accuracy_)

        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
        
sess.close()



