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
import random

import numpy as np
import tensorflow as tf
from network import train_net,test_net
from PIL import Image

SIZE = 1280
WIDTH = 32
HEIGHT = 40
NUM_CLASSES = 26
iterations = 500

SAVER_DIR = "../data/output/train-saver/letters/"

LETTERS_DIGITS = (
"A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
"I", "O")
license_num = ""

time_begin = time.time()

# 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
x = tf.placeholder(tf.float32, shape=[None, SIZE])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])
keep_prob = tf.placeholder(tf.float32)

if __name__ == '__main__' and sys.argv[1] == 'train':
    # 第一次遍历图片目录是为了获取图片总数
    input_count = 0
    for i in range(0 + 10, NUM_CLASSES + 10):
        dir = '../data/license_plate_dt_v1/train_images/training-set/letters/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                input_count += 1

    # 定义对应维数和各维长度的数组
    input_images = np.array([[0] * SIZE for i in range(input_count)])
    input_labels = np.array([[0] * NUM_CLASSES for i in range(input_count)])

    # 第二次遍历图片目录是为了生成图片数据和标签
    index = 0
    for i in range(0 + 10, NUM_CLASSES + 10):
        dir = '../data/license_plate_dt_v1/train_images/training-set/letters/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                        if img.getpixel((w, h)) > 230:
                            input_images[index][w + h * width] = 0
                        else:
                            input_images[index][w + h * width] = 1
                # print ("i=%d, index=%d" % (i, index))
                input_labels[index][i - 10] = 1
                index += 1

    # 第一次遍历图片目录是为了获取图片总数
    val_count = 0
    for i in range(0 + 10, NUM_CLASSES + 10):
        dir = '../data/license_plate_dt_v1/train_images/validation-set/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                val_count += 1

    # 定义对应维数和各维长度的数组
    val_images = np.array([[0] * SIZE for i in range(val_count)])
    val_labels = np.array([[0] * NUM_CLASSES for i in range(val_count)])

    # 第二次遍历图片目录是为了生成图片数据和标签
    index = 0
    for i in range(0 + 10, NUM_CLASSES + 10):
        dir = '../data/license_plate_dt_v1/train_images/validation-set/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                        if img.getpixel((w, h)) > 230:
                            val_images[index][w + h * width] = 0
                        else:
                            val_images[index][w + h * width] = 1
                val_labels[index][i - 10] = 1
                index += 1

    with tf.Session() as sess:
        # get training network
        y_conv = train_net(x_image, keep_prob, NUM_CLASSES)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess.run(tf.global_variables_initializer())

        time_elapsed = time.time() - time_begin
        print("读取图片文件耗费时间：%d秒" % time_elapsed)
        time_begin = time.time()

        print("一共读取了 %s 个训练图像， %s 个标签" % (input_count, input_count))

        # 设置每次训练op的输入个数和迭代次数，这里为了支持任意图片总数，定义了一个余数remainder，譬如，如果每次训练op的输入个数为60，图片总数为150张，则前面两次各输入60张，最后一次输入30张（余数30）
        batch_size = 60
        iterations = iterations
        batches_count = int(input_count / batch_size)
        remainder = input_count % batch_size
        print("训练数据集分成 %s 批, 前面每批 %s 个数据，最后一批 %s 个数据" % (batches_count + 1, batch_size, remainder))

        # 执行训练迭代
        for it in range(iterations):
            # 这里的关键是要把输入数组转为np.array
            for n in range(batches_count):
                train_step.run(feed_dict={x: input_images[n * batch_size:(n + 1) * batch_size],
                                          y_: input_labels[n * batch_size:(n + 1) * batch_size], keep_prob: 0.5})
            if remainder > 0:
                start_index = batches_count * batch_size;
                train_step.run(feed_dict={x: input_images[start_index:input_count - 1],
                                          y_: input_labels[start_index:input_count - 1], keep_prob: 0.5})

            # 每完成五次迭代，判断准确度是否已达到100%，达到则退出迭代循环
            iterate_accuracy = 0
            if it % 5 == 0:
                iterate_accuracy = accuracy.eval(feed_dict={x: val_images, y_: val_labels, keep_prob: 1.0})
                print('第 %d 次训练迭代: 准确率 %0.5f%%' % (it, iterate_accuracy * 100))
                if iterate_accuracy >= 0.9999 and it >= iterations:
                    break;

        print('完成训练!')
        time_elapsed = time.time() - time_begin
        print("训练耗费时间：%d秒" % time_elapsed)
        time_begin = time.time()

        # 保存训练结果
        if not os.path.exists(SAVER_DIR):
            print('不存在训练数据保存目录，现在创建保存目录')
            os.makedirs(SAVER_DIR)
        # 初始化saver
        saver = tf.train.Saver()
        saver_path = saver.save(sess, "%smodel.ckpt" % (SAVER_DIR))

if __name__ == '__main__' and sys.argv[1] == 'predict':
    saver = tf.train.import_meta_graph("%smodel.ckpt.meta" % (SAVER_DIR))
    with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint(SAVER_DIR)
        saver.restore(sess, model_file)
        y_pre = test_net(x_image, keep_prob,sess)
        # 定义优化器和训练op
        conv = tf.nn.softmax(y_pre)
        path = "../data/license_plate_dt_v1/test_images/pletters"
        file_sets = os.listdir(path)
        for item in file_sets:
            img_name = os.path.join(path, item)
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

            license_num += "-"
            license_num = license_num + LETTERS_DIGITS[max1_index]
            print("概率：  [%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (
            LETTERS_DIGITS[max1_index], max1 * 100, LETTERS_DIGITS[max2_index], max2 * 100, LETTERS_DIGITS[max3_index],
            max3 * 100))

        print("城市代号是: 【%s】" % license_num)
