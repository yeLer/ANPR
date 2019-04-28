#!/usr/bin/env python
# encoding: utf-8
'''
@author: lele Ye
@contact: 1750112338@qq.com
@software: pycharm 2018.2
@file: models.py
@time: 2019/4/28 15:39
@desc:
'''
import tensorflow as tf

# 定义卷积函数
def conv_layer(inputs, W, b, conv_strides, kernel_size, pool_strides, padding):
    L1_conv = tf.nn.conv2d(inputs, W, strides=conv_strides, padding=padding)
    L1_relu = tf.nn.relu(L1_conv + b)
    return tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')


# 定义全连接层函数
def full_connect(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)

# 定义整个训练网络
def train_net(x_image,_dropout,NUM_CLASSES):
    # 第一个卷积层
    W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev=0.1), name="W_conv1")
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[16]), name="b_conv1")
    conv_strides = [1, 1, 1, 1]
    kernel_size = [1, 2, 2, 1]
    pool_strides = [1, 2, 2, 1]
    L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')

    # 第二个卷积层
    W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1), name="W_conv2")
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]), name="b_conv2")
    conv_strides = [1, 1, 1, 1]
    kernel_size = [1, 1, 1, 1]
    pool_strides = [1, 1, 1, 1]
    L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')

    # 全连接层
    W_fc1 = tf.Variable(tf.truncated_normal([16 * 20 * 32, 512], stddev=0.1), name="W_fc1")
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]), name="b_fc1")
    h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20 * 32])
    h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)
    # dropout层
    h_fc1_drop = tf.nn.dropout(h_fc1, _dropout)

    W_fc2 = tf.Variable(tf.truncated_normal([512, NUM_CLASSES], stddev=0.1), name="W_fc2")
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name="b_fc2")

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv

def test_net(x_image,keep_prob,sess):
    # 第一个卷积层
    W_conv1 = sess.graph.get_tensor_by_name("W_conv1:0")
    b_conv1 = sess.graph.get_tensor_by_name("b_conv1:0")
    conv_strides = [1, 1, 1, 1]
    kernel_size = [1, 2, 2, 1]
    pool_strides = [1, 2, 2, 1]
    L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')

    # 第二个卷积层
    W_conv2 = sess.graph.get_tensor_by_name("W_conv2:0")
    b_conv2 = sess.graph.get_tensor_by_name("b_conv2:0")
    conv_strides = [1, 1, 1, 1]
    kernel_size = [1, 1, 1, 1]
    pool_strides = [1, 1, 1, 1]
    L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')

    # 全连接层
    W_fc1 = sess.graph.get_tensor_by_name("W_fc1:0")
    b_fc1 = sess.graph.get_tensor_by_name("b_fc1:0")
    h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20 * 32])
    h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)

    # dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout层
    W_fc2 = sess.graph.get_tensor_by_name("W_fc2:0")
    b_fc2 = sess.graph.get_tensor_by_name("b_fc2:0")

    y_pre = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_pre