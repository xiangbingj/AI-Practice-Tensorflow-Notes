#coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 导入 MNIST 数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)
# 查看数据集形状
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
 
for i in range(10):
    plt.subplot(2,5,i+1)
    # 显示图片 imshow
    plt.imshow(images[i].reshape(l_,l_))
    # 添加标题为对应的label
    # np.where 返回对应的索引
    plt.title(int(np.where(labels[i]==1)[0]))
    # 保存 imsave
    plt.imsave('./data/images_'+str(i+1)+'.jpg', images[i].reshape((l_,l_)))
plt.show()
# 读取保存的图片 imread
img = plt.imread('./data/images_1.jpg')
 
# 查看形状
# （28， 28， 4）
# 发现有4通道
# 第4通道的值全部为255
print(img.shape)
print('===' * 30)
print(img.T[:][:][2])

# 第2种保存方式，保存后图片仍为原始尺寸
 
from scipy import misc
 
 
l = len(mnist.train.images[1])
# 开平方
l_ = int(np.sqrt(l))
 
images = mnist.train.images
labels = mnist.train.labels
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(images[i].reshape(l_,l_))
    plt.title(int(np.where(labels[i]==1)[0]))
    misc.imsave('./data/images_'+str(i+10)+'.jpg', images[i].reshape((l_,l_)))
plt.show()

# 查看保存的结果
img = plt.imread('./data/images_11.jpg')
print(img.shape)

