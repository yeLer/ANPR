#!/usr/bin/env python3
# encoding: utf-8
"""
@project: pythonWorkSpace
@time: 2018/7/20 19:32
@author: yeLer082
@contact: 1750112338@qq.com
@desc:
"""
import sys
import os
import time
import numpy as np
import tensorflow as tf
sys.path.append('..')
from network import train_net
from PIL import Image

SIZE = 1280
WIDTH = 32
HEIGHT = 40
NUM_CLASSES = 34
iterations = 1000

SAVER_DIR = "../data/output/train-saver/digits/"

LETTERS_DIGITS = (
"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P",
"Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")
license_num = ""

time_begin = time.time()

# 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
x = tf.placeholder(tf.float32, shape=[None, SIZE])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])
keep_prob = tf.placeholder(tf.float32)

if __name__ == '__main__' and sys.argv[1] == 'predict':
    saver = tf.train.import_meta_graph("%smodel.ckpt.meta" % (SAVER_DIR))
    with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint(SAVER_DIR)
        saver.restore(sess, model_file)

        y_pre = test_net(x_image, keep_prob, sess)
        # 定义优化器和训练op
        conv = tf.nn.softmax(y_pre)
        path = "../data/license_plate_dt_v1/test_images/letters"
        file_sets = os.listdir(path)
        for item in file_sets:
            img_name = os.path.join(path,item)
            img = Image.open(img_name)
            width = img.size[0]
            height = img.size[1]

            img_data = [[0] * SIZE for i in range(1)]
            for h in range(0, height):
                for w in range(0, width):
                    if img.getpixel((w, h)) < 190:
                        img_data[0][w + h * width] = 1
                    else:
                        img_data[0][w + h * width] = 0

            result = sess.run(conv, feed_dict={x: np.array(img_data), keep_prob: 1.0})

            max1 = 0
            max2 = 0
            max3 = 0
            max1_index = 0
            max2_index = 0
            max3_index = 0
            for j in range(NUM_CLASSES):
                if result[0][j] > max1:
                    max1 = result[0][j]
                    max1_index = j
                    continue
                if (result[0][j] > max2) and (result[0][j] <= max1):
                    max2 = result[0][j]
                    max2_index = j
                    continue
                if (result[0][j] > max3) and (result[0][j] <= max2):
                    max3 = result[0][j]
                    max3_index = j
                    continue

            license_num = license_num + LETTERS_DIGITS[max1_index]
            print(item+"概率：  [%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (
            LETTERS_DIGITS[max1_index], max1 * 100, LETTERS_DIGITS[max2_index], max2 * 100, LETTERS_DIGITS[max3_index],
            max3 * 100))

        print("车牌编号是: 【%s】" % license_num)
