#coding: utf-8

import h5py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 导入 MNIST 数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
print(mnist.test.images.shape)
print(mnist.test.labels.shape)
# 打印前10张图片
 
l = len(mnist.train.images[1])
# 开平方  l_为28
l_ = int(np.sqrt(l))
 
images = mnist.train.images
# 标签
labels = mnist.train.labels
 
# 查看数据集形状
# 第3种保存方式

 
im_index = np.zeros((10, 28,28,1))
la_index = np.zeros((10,10))
for i in range(10):
    f = h5py.File('train.h5', 'w')
    im_index[i,:] = images[i].reshape((28,28,1))
    la_index[i,:] = labels[i]
    f['data'] = im_index
    f['label'] = la_index
    f.close()

rf = h5py.File('train.h5')
print(rf['data'])
print(rf['label'])
print(rf['data'].shape)
print(rf['label'].shape)
