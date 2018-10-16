#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 22:59:26 2018

@author: makise
"""

import os
import tensorflow as tf
import tensorlayer as tl
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.python.client import device_lib
import time
#import imutils
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "99"
 
#print(device_lib.list_local_devices())
#XX_train, yy_train, XX_test, yy_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)



def load_scan(path,name):
    
    all_files = []
    
    for dirName, subdirList, fileList in os.walk(path):

        for filename in fileList:

            if name in filename.lower():

                all_files.append(os.path.join(dirName,filename))

    return all_files 
 
 
class_path = "/home/makise/DeepLearning/Company/Data/"
classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}  # 

#ImageFileName = "MyTrainData"

#aaa = load_scan(image_path,"airplane")
def CalcImgCounts(class_path,classes,ImageFileName = ""):
    
    image_path = class_path + ImageFileName + '/'
    
    ImgCounts = 0
    
    for index,name in enumerate(classes):
             
        Counts = len(load_scan(image_path, name))
        
        ImgCounts += Counts
        
        print(name, ImgCounts)

    return(ImgCounts)

TrainImgCounts = CalcImgCounts(class_path = class_path, classes = classes ,ImageFileName = "MyTrainData" )
TestImgCounts = CalcImgCounts(class_path = class_path, classes = classes ,ImageFileName = "MyTestData" )

#len(aaa)

def data_to_tfrecord(class_path,classes,ImageFileName = "",RecordFileName = "",OutputFileName = "Default.tfrecords"):
    
#    ImageFileName = 'MyTrainData'
    
#    class_path + RecordFileName + 'train.tfrecords'
    image_path = class_path + ImageFileName + '/'
    
    writer = tf.python_io.TFRecordWriter(class_path+RecordFileName+OutputFileName)
    
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
     
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
 
    for index, name in enumerate(classes):
        
    #    name = 'MyTrain'
#        name = 'dog'
    #    class_path = cwd
    
        print(image_path,name,index)
        
        for img_name in load_scan(image_path, name):
            
#            img_name = '/home/makise/DeepLearning/Company/Data/MyTrainData/4878_dog.png'
            
            img_path = img_name  
            
            img = Image.open(img_path)
            
            img = img.resize((32, 32))

            img = np.array(img).astype("float32")
#            
            img_raw = img.tobytes()  
            
            index = int(index)
            
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": _int64_feature(index),
                "img_raw": _bytes_feature(img_raw),
            }))
            writer.write(example.SerializeToString())  
            
    writer.close()
    print("writed OK")

data_to_tfrecord(class_path = class_path,ImageFileName = 'MyTrainData' ,classes = classes,OutputFileName = 'train.tfrecords')
data_to_tfrecord(class_path = class_path,ImageFileName = 'MyTestData' ,classes = classes,OutputFileName = 'test.tfrecords')

#TFRecordName = 'train.tfrecords'
def read_and_decode(class_path,RecordFileName = "",TFRecordName = "Default.tfrecords",is_train = None ):  # read train.tfrecords
    
    if RecordFileName == "":
#        TFRecordName = 'train.tfrecords'
        filename = class_path+TFRecordName
        
    else:
         
        filename = class_path+RecordFileName+'/'+TFRecordName
    
    filename_queue = tf.train.string_input_producer([filename])  
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  
    features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw': tf.FixedLenFeature([], tf.string),
                                   }) 
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [32, 32, 3])  
    
    if is_train == True:
        
        img = tf.random_crop(img, [24,24,3])
        
        img = tf.image.random_flip_left_right(img)
        
        img = tf.image.random_brightness(img, max_delta = 63)
        
        img = tf.image.random_contrast(img, lower = 0.2, upper = 1.8)
        
        img = tf.image.per_image_standardization(img)
        
    elif is_train == False:
        
        img = tf.image.resize_image_with_crop_or_pad(img,24,24)
        
        img = tf.image.per_image_standardization(img)
        
    elif is_train == None:
        
        img = img
        
    label = tf.cast(features["label"], tf.int32)
    
    return(img, label)
    
    
batch_size = 50

#for example in tf.python_io.tf_record_iterator("/home/makise/DeepLearning/Company/Data/train.tfrecords"):
#    result = tf.train.Example.FromString(example)
#    
#for example in tf.python_io.tf_record_iterator("/home/makise/train.cifar10"):
#    result = tf.train.Example.FromString(example)

#sess = tf.Session()
#class_path = '/home/makise/'
with tf.device('/cpu:0'):
    
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True))
    
    x_train, y_train = read_and_decode(class_path = class_path, TFRecordName = 'train.tfrecords',is_train = True)
    
    x_test,y_test = read_and_decode(class_path = class_path, TFRecordName = 'test.tfrecords', is_train = False)
    
    x_train_batch, y_train_batch = tf.train.shuffle_batch([x_train, y_train], batch_size = batch_size, capacity = 2000, min_after_dequeue = 1000, num_threads = 32)
    
    x_test_batch, y_test_batch = tf.train.batch([x_test, y_test], batch_size = batch_size, capacity = 3000, num_threads = 32)

#    sess.run(print(x_train))
    
def model(x_crop, y_label, reuse = None, is_train = None):
    
    W_init = tf.truncated_normal_initializer(stddev = 5e-2)
    
    W_init2 = tf.truncated_normal_initializer(stddev = 0.04)
    
    b_init2 = tf.constant_initializer(value = 0.1)
    
    with tf.variable_scope("model", reuse = reuse):
        
        net = tl.layers.InputLayer(x_crop, name = 'input')
        
        net = tl.layers.Conv2d(net, n_filter = 64, filter_size = (5,5), strides = (1,1), padding = 'SAME', W_init = W_init, b_init = None , name = 'cnn1')
        
        net = tl.layers.BatchNormLayer(net, is_train, act = tf.nn.relu, name = 'bn1')
        
        net  = tl.layers.MaxPool2d(net, filter_size = (3,3), strides = (2,2), padding = 'SAME', name = 'p1')
        
        net =  tl.layers.Conv2d(net, n_filter = 64, filter_size = (5,5), strides = (1,1), padding = 'SAME', W_init = W_init, b_init = None, name = 'cnn2')
        
        net = tl.layers.BatchNormLayer(net, is_train, act = tf.nn.relu, name = 'bn2')
        
        net = tl.layers.MaxPool2d(net, filter_size = (3,3), strides = (2,2), padding = 'SAME', name = 'p2')
        
        net = tl.layers.FlattenLayer(net, name = 'flatten')
        
        net = tl.layers.DenseLayer(net, n_units = 384, act = tf.nn.relu, W_init = W_init2, b_init = b_init2, name = 'd1relu')
        
        net = tl.layers.DenseLayer(net, n_units = 192, act = tf.nn.relu, W_init = W_init2, b_init = b_init2, name = 'd2relu')
        
        net = tl.layers.DenseLayer(net, n_units = 10, act = tf.identity, name = 'output')
        
        output =net.outputs
        
        ce = tl.cost.cross_entropy(output,y_label, name = 'cost')
        
        L2 = 0
        
        for p in tl.layers.get_variables_with_name('relu/W', True, True):
            
            L2 = L2 + tf.contrib.layers.l2_regularizer(0.004)(p)
            
        cost = ce + L2
        
        accuracy = tf.metrics.accuracy(labels = y_label, predictions = tf.argmax(output, axis = 1),)[1]

        return(net, cost, accuracy)

#with tf.device('/gpu:0'):
#    network, cost, acc = model(x_train_batch, y_train_batch, reuse = tf.AUTO_REUSE, is_train = True)
#
#    testnetwork, Testcost, Testacc = model(x_test_batch, y_test_batch, reuse = tf.AUTO_REUSE, is_train = False)

#with tf.device('/gpu:0'):
network, cost, acc = model(x_train_batch, y_train_batch, reuse = tf.AUTO_REUSE, is_train = True)

testnetwork, Testcost, Testacc = model(x_test_batch, y_test_batch, reuse = tf.AUTO_REUSE, is_train = False)

network.print_params(False)
network.print_layers()


n_epoch = 5000

learning_rate = 0.0001

print_freq = 1

n_step_epoch = int(TrainImgCounts/batch_size)
#n_step_epoch = int(len(yy_train)/batch_size)

n_step = n_epoch * n_step_epoch

train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op

sess.run(init_op)

coord = tf.train.Coordinator()

threads = tf.train.start_queue_runners(sess = sess, coord = coord)

step = 0

for epoch in range(0,n_epoch):
    
    start_time = time.time()
    
    train_loss, train_acc,n_batch = 0, 0, 0
    
    for s in range(0,n_step_epoch):
        
        err, ac, _ = sess.run([cost, acc, train_op])
        
        step += 1
        
        train_loss += err
        
        train_acc += ac
        
        n_batch += 1
        
    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print(
                "Epoch %d : Step %d-%d of %d took %fs" %
                (epoch, step, step + n_step_epoch, n_step, time.time() - start_time)
            )
            print("   train loss: %f" % (train_loss / n_batch))
            print("   train acc: %f" % (train_acc / n_batch))

            test_loss, test_acc, n_batch = 0, 0, 0
            for _ in range(int(TestImgCounts / batch_size)):
#            for _ in range(int(len(yy_test) / batch_size)):
                err, ac = sess.run([Testcost, Testacc])
                test_loss += err
                test_acc += ac
                n_batch += 1
            print("   test loss: %f" % (test_loss / n_batch))
            print("   test acc: %f" % (test_acc / n_batch))

#    if (epoch + 1) % (print_freq * 50) == 0:
#        print("Save model " + "!" * 10)
#        saver = tf.train.Saver()
#        save_path = saver.save(sess, model_file_name)
#        # you can also save model into npz
#        tl.files.save_npz(network.all_params, name='model.npz', sess=sess)
#        # and restore it as follow:
#        # tl.files.load_and_assign_npz(sess=sess, name='model.npz', network=network)
            
for step in range(0,1200):
    
    err, ac, _ = sess.run([cost, acc, train_op])
    
    if step % 50 == 0:
        
        test_loss, test_acc = sess.run([Testcost, Testacc])
        print('train loss: %.4f' % err, '| train accuracy: %.2f' % ac)
        
        print('test loss: %.4f' % test_loss, '| test accuracy: %.2f' % test_acc)

        
            
            


coord.request_stop()
coord.join(threads)
sess.close()
        
    
        
        
        
    
    
