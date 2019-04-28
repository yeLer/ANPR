#!/usr/bin/env python
# encoding: utf-8
'''
@author: lele Ye
@contact: 1750112338@qq.com
@software: pycharm 2018.2
@file: lo.py
@time: 2019/4/27 17:38
@desc:
'''
import cv2

# 使用的是HyperLPR已经训练好了的分类器
watch_cascade = cv2.CascadeClassifier('./model/detector_ch.xml')

# 先读取图片
image = cv2.imread("../data/demo/000003.jpg")

resize_h = 1000
height = image.shape[0]
scale = image.shape[1]/float(image.shape[0])
image = cv2.resize(image, (int(scale*resize_h), resize_h))

image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
watches = watch_cascade.detectMultiScale(image_gray, 1.1, 2, minSize=(36, 9), maxSize=(36*40, 9*40))

print("检测到车牌数", len(watches))
for (x, y, w, h) in watches:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()