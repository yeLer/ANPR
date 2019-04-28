import os

import cv2
import numpy as np

minPlateRatio = 0.5  # 车牌最小比例
maxPlateRatio = 6  # 车牌最大比例
# read lower_col_hsv from hmin,smin,vmin
# read higher_col_hsv from hmax,smax,vmax
# 定义蓝底车牌的hsv颜色区间
lower_blue_col_hsv = np.array([100, 40, 50])
higher_blue_col_hsv = np.array([140, 255, 255])

# 定义黄底车牌的hsv颜色区间
lower_yellow_col_hsv = np.array([11, 43, 46])
higher_yellow_col_hsv = np.array([34, 255, 255])

# 定义黑底车牌的hsv颜色区间
lower_black_col_hsv = np.array([0, 0, 0])
higher_black_col_hsv = np.array([180, 255, 46])

# 定义白底车牌的hsv颜色区间
lower_white_col_hsv = np.array([0, 0, 46])
higher_white_col_hsv = np.array([180, 30, 220])

# 定义绿底车牌的hsv颜色区间
lower_green_col_hsv = np.array([35, 43, 46])
higher_green_col_hsv = np.array([77, 255, 255])


# 找到符合车牌形状的矩形
def findPlateNumberRegion(img):
    region = []
    # 查找外框轮廓
    contours_img, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("contours lenth is :%s" % (len(contours)))
    # 筛选面积小的
    list_rate = []
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算轮廓面积
        area = cv2.contourArea(cnt)
        # 面积小的忽略
        if area < 1000:
            continue
        # 转换成对应的矩形（最小）
        rect = cv2.minAreaRect(cnt)
        # print("rect is:%s" % {rect})
        # 根据矩形转成box类型，并int化
        box = np.int32(cv2.boxPoints(rect))
        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        # 正常情况车牌长高比在2.7-5之间,那种两行的有可能小于2.5，这里不考虑
        ratio = float(width) / float(height)
        rate = getxyRate(cnt)
        print("area", area, "ratio:", ratio, "rate:", rate)
        if ratio > maxPlateRatio or ratio < minPlateRatio:
            continue
        # 符合条件，加入到轮廓集合
        region.append(box)
        list_rate.append(ratio)
    index = getSatifyestBox(list_rate)
    return region[index]

#找出最有可能是车牌的位置
def getSatifyestBox(list_rate):
    for index, key in enumerate(list_rate):
        list_rate[index] = abs(key - 3)
    print(list_rate)
    index = list_rate.index(min(list_rate))
    print(index)
    return index


def getxyRate(cnt):
    x_height = 0
    y_height = 0
    x_list = []
    y_list = []
    for location_value in cnt:
        location = location_value[0]
        x_list.append(location[0])
        y_list.append(location[1])
    x_height = max(x_list) - min(x_list)
    y_height = max(y_list) - min(y_list)
    return x_height * (1.0) / y_height * (1.0)


def location(file):
    img = cv2.imread(file)
    # 转换成hsv模式图片
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 找到hsv图片下的所有符合蓝黄黑白绿五种颜色区间的像素点，转换成二值化图像
    mask_blue = cv2.inRange(hsv_img, lower_blue_col_hsv, higher_blue_col_hsv)
    mask_yellow = cv2.inRange(hsv_img, lower_yellow_col_hsv, higher_yellow_col_hsv)
    mask_black = cv2.inRange(hsv_img, lower_black_col_hsv, higher_black_col_hsv)
    mask_white = cv2.inRange(hsv_img, lower_white_col_hsv, higher_white_col_hsv)
    mask_green = cv2.inRange(hsv_img, lower_green_col_hsv, higher_green_col_hsv)
    res = cv2.bitwise_and(img, img, mask=mask_blue+mask_yellow+mask_black+mask_white+mask_green)

    # 灰度化
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # 高斯模糊：车牌识别中利用高斯模糊将图片平滑化，去除干扰的噪声对后续图像处理的影响
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    #sobel算子：车牌定位的核心算法，水平方向上的边缘检测，检测出车牌区域
    sobel = cv2.convertScaleAbs(cv2.Sobel(gaussian, cv2.CV_16S, 1, 0, ksize=3))
    #进一步对图像进行处理，强化目标区域，弱化背景。
    ret, binary = cv2.threshold(sobel, 150, 255, cv2.THRESH_BINARY)


    # 进行闭操作，闭操作可以将目标区域连成一个整体，便于后续轮廓的提取
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)
    # 进行开操作，去除细小噪点
    eroded = cv2.erode(closed, None, iterations=1)
    dilation = cv2.dilate(eroded, None, iterations=1)

    # 查找并筛选符合条件的矩形区域
    region = findPlateNumberRegion(closed)
    cv2.drawContours(img, [region], 0, (0, 255, 0), 2)
    # 将符合条件的区域保存

    cv2.imshow("img", img)
    cv2.waitKey(0)

def test_mul_imgs(path):
    fileList = os.listdir(path)
    for file in fileList:
        try:
            location(path + "/" + file)
        except:
            print("异常:", file)


if __name__ == '__main__':
    file = r"../data/demo/000003.jpg"
    location(file)