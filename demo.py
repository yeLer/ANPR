#!/usr/bin/env python
# encoding: utf-8
'''
@author: lele Ye
@contact: 1750112338@qq.com
@software: pycharm 2018.2
@file: demo.py
@time: 2018/9/26 15:41
@desc:
'''
#导入hyperlpr包
from hyperlpr import *
#导入OpenCV库
import cv2
import argparse
import os
from PIL import Image,ImageDraw,ImageFont
import numpy as np
from utils.nms.py_cpu_nms import py_cpu_nms
#读入图片
def recognition(args):
    image = cv2.imread(args.filename)
    re_list = HyperLPR_PlateRecogntion(image)  # 识别结果
    cls_boxes = []
    cls_scores = []
    if len(re_list)>0:
        for item in re_list:
            score, box = item[1],item[2]
            cls_scores.append(score)
            cls_boxes.append(box)
        cls_boxes = np.array(cls_boxes)
        cls_scores = np.array(cls_scores)
        dets = np.hstack((cls_boxes,cls_scores[:,np.newaxis])).astype(np.float32)
        NMS_THREASH = 0.45
        det_re = py_cpu_nms(dets,NMS_THREASH)
        re_list = [re_list[x] for x in det_re]
    for item in re_list:
        label,probolity,position = item[0],item[1],item[2]
        cv2.rectangle(image, (position[0],position[1]),(position[2],position[3]),(0,255,0),2)
        position = (position[0],position[1])
        image = draw_chinese(image,position,label)
        # cv2.putText(image,label,(position[0],position[1]+10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,2,255),2)
    image_name = os.path.basename(args.filename)
    cv2.imwrite((args.save_dir+"/"+image_name),image)

def draw_chinese(img_OpenCV,position,label):
    '''
    :param img_OpenCV: 读取到的opencv格式的图片
    :param position: 写文本的位置
    :param label: 在图片上显示的内容
    :return:
    '''
    # 图像从OpenCV格式转换成PIL格式
    img_PIL = Image.fromarray(cv2.cvtColor(img_OpenCV, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('./fonts/fangzheng_heiti.TTF', 28)
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, label, font=font, fill=(255, 0, 0))
    # 转换回OpenCV格式
    img_OpenCV = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    # cv2.imshow("print chinese to image", img_OpenCV)
    return img_OpenCV


def parser_args():
    parser = argparse.ArgumentParser(description='hyperlpr on license recognition')
    parser.add_argument('--name',dest='filename',help='testing file name',
                        default='./data/demo/000004.jpg')
    parser.add_argument('--save', dest='save_dir', help='save file dir',
                        default='./data/results')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parser_args()
    recognition(args)