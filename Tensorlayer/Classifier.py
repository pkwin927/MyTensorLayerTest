#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 01:20:04 2018

@author: makise
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 05:28:31 2018

@author: makise
"""

import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt
import numpy as np

tf.reset_default_graph()

tf.set_random_seed(1)
np.random.seed(1)

n_data = np.ones((100, 2))
x0 = np.random.normal(5 * n_data,1)      # class0 x shape=(100, 2)
y0 = np.zeros(100)                      # class0 y shape=(100, 1)
x1 = np.random.normal(-5*n_data, 1)     # class1 x shape=(100, 2)
y1 = np.ones(100)                       # class1 y shape=(100, 1)
x2 = np.random.normal(0*n_data,1)
y2 = np.full((100,),2)
x = np.vstack((x0, x1, x2))  # shape (300, 3) + some noise
y = np.hstack((y0, y1 ,y2))  # shape (300, )

# plot data
plt.scatter(x[:, 0], x[:, 1], c=y, s=100, lw=0, cmap='RdYlGn')
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)

tf_y = tf.placeholder(tf.int32, y.shape)

#layer1 = tf.layers.dense(tf_x, units = 10,activation= tf.nn.relu)
#reuse = True
#
#def mlp(tf_x, reuse = True):
#    

def model(tf_x, is_train = True, reuse = tf.AUTO_REUSE):
    
    with tf.variable_scope("model", reuse = reuse):
            
    #    tl.layers.set_name_reuse(reuse)
    
        Input = tl.layers.InputLayer(tf_x, name = 'Input')
        
        layer1 = tl.layers.DenseLayer(Input, n_units = 10, act = tf.nn.relu, name = "layer1")
        
        layer2 = tl.layers.DenseLayer(layer1, n_units = 100, act = tf.nn.relu,name = "layer2")
        #layer2 = tf.layers.dense(layer1, units = 100,activation= tf.nn.relu)
        
        layer3 = tl.layers.DenseLayer(layer2, n_units = 3, name = "OutPut")
        
        #output = tf.layers.dense(layer2, 3)
    return(layer3)

network = model(tf_x, is_train = False, reuse = tf.AUTO_REUSE) 

output = network.outputs
#return(output)

#output = mlp(tf_x, reuse = True)

#Output = model(tf_x, is_train = False, reuse = )

loss = tf.losses.sparse_softmax_cross_entropy(labels = tf_y, logits = output)

#accuracy = tf.metrics.accuracy()

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf_y, predictions=tf.argmax(output, axis=1),)[1]
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_op = optimizer.minimize(loss)

sess = tf.Session()

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

sess.run(init_op)


plt.ion()   # something about plotting
for step in range(1000):
    # train and net output
    
    i, acc, pred = sess.run([train_op, accuracy, output], {tf_x: x, tf_y: y})
    if step % 50 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x[:, 0], x[:, 1], c=pred.argmax(1), s=100, lw=0, cmap='RdYlGn')
        plt.text(1.5, -4, 'Accuracy=%.2f' % acc, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()















 