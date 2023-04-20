import time

import cv2
import numpy as np
from skimage import morphology
from matplotlib import pyplot as plt

imaPath = 'normal/34.png'
maskPath = 'postprocessed/Result_0.png'
rect = [{'top': 855, 'left': 564, 'bottom': 959, 'right': 644},
        {'top': 1028, 'left': 402, 'bottom': 1304, 'right': 600}]
# mid_x0 = (rect[0]['left'] + rect[0]['right']) // 2
# mid_x1 = (rect[1]['left'] + rect[1]['right']) // 2
y0 = rect[0]['top'] + 2 * (rect[0]['bottom'] - rect[0]['top']) // 3
# y1 = rect[1]['top'] + 2 * (rect[1]['bottom'] - rect[1]['top']) // 3
y1 = rect[1]['top'] + (rect[1]['bottom'] - rect[1]['top']) // 2
y_qc = [y0, y1]


#########################################################
# 获取质控区掩膜
#########################################################
def getQcMask():
    img = cv2.imread(imaPath)
    mask = cv2.imread(maskPath)
    mask_img = cv2.bitwise_and(img, img, mask=mask[:, :, 0])
    qcmask = np.zeros(img.shape[:2], dtype="uint8")
    qc = []
    for rec in rect:
        cv2.rectangle(qcmask, (rec['left'], rec['top']), (rec['right'], rec['bottom']), 255, -1)
        # qc_img = cv2.bitwise_and(mask_img, mask_img, mask=qcmask)   #原图上的掩膜
        # qc_img = cv2.cvtColor(qc_img, cv2.COLOR_BGR2GRAY)
        # 质控区图像列表
        temp = mask_img[rec['top']:rec['bottom'], rec['left']:rec['right']]
        qc.append(cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY))
    return mask_img, qc


#########################################################
# 质控点轮廓内切圆
#########################################################
def preprocess(qc):
    # mask_qcimg = cv2.threshold(qc_img, 1, 255, cv2.THRESH_BINARY)[1]
    # 二值化
    mask_qc = []
    for img in qc:
        mask_qc.append(cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1])
    return mask_qc


# 识别最大轮廓
def getcontour():
    mask = cv2.imread(maskPath)
    mask = mask[:, :, 0]
    contour = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    area = []
    for i in range(len(contour)):
        area.append(cv2.contourArea(contour[i]))
    contour = contour[np.argmax(area)]
    return contour


#########################################################
# 中心线法找内接圆
#########################################################
# 找到中心线
def getSkeleton(binary):
    binary = binary / 255
    skeleton0 = morphology.skeletonize(binary)
    skeleton = skeleton0.astype(np.uint8) * 255
    return skeleton


# 找到质控线与中心线交点，画出质控圆
# def getCircle(mask_qc, contours, skeletons):
#     result = cv2.bitwise_xor(mask_qc[0], skeletons[0])
#     pt = (0, 0)
#     pre = 2000
#     for i in range(skeletons[0].shape[1]):  # 宽
#         if skeletons[0][y0, i] == 255:
#             if abs(i - mid_x0) < pre:
#                 pt = (i, y0)
#                 pre = abs(i - mid_x0)
#     radius = abs(cv2.pointPolygonTest(contours[0], pt, True))  # (x, y) x:x轴坐标 y:y轴坐标
#     radius = int(radius)
#     print(radius)
#     cv2.circle(mask_qc[0], pt, radius, 0)
#     cv2.namedWindow('mask_qc', 0)
#     cv2.imshow('mask_qc', mask_qc[0])
#     cv2.waitKey(0)
#     # cv2.circle(mask_qc[0], pt, 1, (0, 0, 255), -1)
#     # cv2.namedWindow('mask_qc', 0)
#     # cv2.imshow('mask_qc', mask_qc[0])
#     # cv2.waitKey(0)
#     # print(pt)
#     return pt, radius

#########################################################
# 最大半径法找内接圆
#########################################################
def getCircle(binary_img, contour):
    radius = []
    centers = []
    for i, y in enumerate(y_qc):
        maxdist = 0
        pt = (0, 0)
        for j in range(rect[i]['left'], rect[i]['right']):
            dist = cv2.pointPolygonTest(contour, (j, y), True)
            if dist > maxdist:
                x0 = j
                y0 = y
                r = dist
                img = np.asarray(binary_img)
                xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
                mask = (xx - x0) ** 2 + (yy - y0) ** 2 < r ** 2
                min_intensity = np.min(img[mask])
                if min_intensity > 0:
                    maxdist = dist
                    pt = (j, y)
        radius.append(int(maxdist))
        centers.append(pt)
    return radius, centers


# 半径：maxVal  圆心：maxDistPt
# 转换格式
def drawCircle(gray_img, radius, centers):
    # 绘制内切圆
    gray_img = np.asarray(gray_img)
    qcpoint = gray_img.copy()
    for i in range(len(radius)):
        cv2.circle(qcpoint, centers[i], radius[i], 255, -1)

    return qcpoint


#########################################################
# 求平均灰度
#########################################################
def getAvgGray(gray_img, radius, centers):
    top, bottom, left, right = [], [], [], []
    cnts = []
    for m in range(len(radius)):
        top.append(centers[m][1] - radius[m])
        bottom.append(centers[m][1] + radius[m])
        left.append(centers[m][0] - radius[m])
        right.append(centers[m][0] + radius[m])
        circle = np.empty(gray_img.shape, dtype="uint8")
        cv2.circle(circle, centers[m], radius[m], 255, -1)
        cnt = cv2.findContours(circle, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts.append(cnt)
    gray_avg = []
    for m in range(len(cnts)):
        gray_total = 0
        count = 0
        for i in range(top[m], bottom[m]):  # 高，y轴
            for j in range(left[m], right[m]):  # 宽，x轴
                dist = cv2.pointPolygonTest(cnts[m][0], (j, i), True)
                if dist > 0:
                    gray_total += gray_img[i, j]
                    count += 1
        gray_avg.append(gray_total // count)
    print(gray_avg)

    # 求四个扇形平均灰度
    # 左上
    gray_avg1 = []
    for m in range(len(cnts)):
        gray_total = 0
        count = 0
        for i in range(top[m], bottom[m]):  # 高，y轴
            for j in range(left[m], right[m]):  # 宽，x轴
                dist = cv2.pointPolygonTest(cnts[m][0], (j, i), True)
                if dist > 0 and j < centers[m][0] and i < centers[m][1]:
                    gray_total += gray_img[i, j]
                    count += 1
        gray_avg1.append(gray_total // count)
    print(gray_avg1)

    # 左下
    gray_avg2 = []
    for m in range(len(cnts)):
        gray_total = 0
        count = 0
        for i in range(top[m], bottom[m]):  # 高，y轴
            for j in range(left[m], right[m]):  # 宽，x轴
                dist = cv2.pointPolygonTest(cnts[m][0], (j, i), True)
                if dist > 0 and j < centers[m][0] and i > centers[m][1]:
                    gray_total += gray_img[i, j]
                    count += 1
        gray_avg2.append(gray_total // count)
    print(gray_avg2)

    # 右上
    gray_avg3 = []
    for m in range(len(cnts)):
        gray_total = 0
        count = 0
        for i in range(top[m], bottom[m]):  # 高，y轴
            for j in range(left[m], right[m]):  # 宽，x轴
                dist = cv2.pointPolygonTest(cnts[m][0], (j, i), True)
                if dist > 0 and j > centers[m][0] and i < centers[m][1]:
                    gray_total += gray_img[i, j]
                    count += 1
        gray_avg3.append(gray_total // count)
    print(gray_avg3)

    # 右下
    gray_avg4 = []
    for m in range(len(cnts)):
        gray_total = 0
        count = 0
        for i in range(top[m], bottom[m]):  # 高，y轴
            for j in range(left[m], right[m]):  # 宽，x轴
                dist = cv2.pointPolygonTest(cnts[m][0], (j, i), True)
                if dist > 0 and j > centers[m][0] and i > centers[m][1]:
                    gray_total += gray_img[i, j]
                    count += 1
        gray_avg4.append(gray_total // count)
    print(gray_avg4)


# mask_img:原图提取的血管区域图像
# qc:两个指控区域图像
# mask_qc:两个指控区域二值图
# skeleton:血管中心线
# contour:血管轮廓
# radius:半径
# centers:圆心
if __name__ == '__main__':
    mask_img, qc = getQcMask()
    gray_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)[1]
    mask_qc = preprocess(qc)

    start = time.time()
    skeleton = getSkeleton(binary_img)
    end = time.time()
    print("skeleton:", (end - start), 's')

    # skeletons = [skeleton[rect[0]['top']:rect[0]['bottom'], rect[0]['left']:rect[0]['right']],
    #              skeleton[rect[1]['top']:rect[1]['bottom'], rect[1]['left']:rect[1]['right']]]
    contour = getcontour()

    start = time.time()
    radius, centers = getCircle(binary_img, contour)
    # print(radius, centers)
    end = time.time()
    print("getCircle:", (end - start), 's')

    start = time.time()
    avg_gray = getAvgGray(gray_img, radius, centers)
    end = time.time()
    print("getGray:", (end - start), 's')

    # qcpoint = drawCircle(gray_img, radius, centers)
    # plt.subplot(121), plt.imshow(gray_img, cmap='gray')
    # plt.title('origin_img'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(qcpoint, cmap='gray')
    # plt.title('qcpoint_img'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # cv2.imwrite('qcpoint.png', qcpoint)
    # 绘制中心线
    # res = cv2.bitwise_xor(binary, skeleton)