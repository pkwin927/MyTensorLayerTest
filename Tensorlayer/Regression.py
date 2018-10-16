#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 02:26:11 2018

@author: makise
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 05:20:36 2018

@author: makise
"""

import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt
import numpy as np


tf.reset_default_graph()

tf.set_random_seed(1)
np.random.seed(1)

x = np.linspace(-2,2,1000)[:, np.newaxis]

noise = np.random.normal(0,0.1, size = x.shape)

y = np.power(x,3) + noise  # y = x^2 + noise

plt.scatter(x, y)
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)

tf_y = tf.placeholder(tf.float32, y.shape)

def model(tf_x, is_train = True, reuse = tf.AUTO_REUSE):
    
    with tf.variable_scope('model', reuse = reuse):
        
        
        inputs = tl.layers.InputLayer(tf_x, name = 'Input')
        
        layer1 = tl.layers.DenseLayer(inputs, 10, act = tf.nn.relu, name = 'layer1')
        
        layer2 = tl.layers.DenseLayer(layer1,  50, act = tf.nn.relu, name = 'layer2')
        
        layer3 = tl.layers.DenseLayer(layer2,1, name = 'layer3')

    return(layer3)

network = model(tf_x, is_train = False, reuse = tf.AUTO_REUSE)

output = network.outputs

loss = tf.losses.mean_squared_error(tf_y, output)   
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)


sess = tf.Session()                                 
sess.run(tf.global_variables_initializer())         

plt.ion()   

for step in range(1000):

    i, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})
    if step % 50 == 0:

        plt.cla()
        plt.scatter(x, y)
        plt.plot(x, pred, 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

sess.close()







