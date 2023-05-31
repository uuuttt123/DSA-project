import math
import time

import cv2
import numpy as np
from skimage import morphology
from matplotlib import pyplot as plt

imaPath = 'normal/34.png'
maskPath = 'postprocessed/Result_0.png'
rect = [{'top': 855, 'left': 564, 'bottom': 959, 'right': 644},
        {'top': 1028, 'left': 402, 'bottom': 1304, 'right': 600}]
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
    contour = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    area = []
    for i in range(len(contour)):
        area.append(cv2.contourArea(contour[i]))
    contour = contour[np.argmax(area)]

    # # 创建一个空白图像，用于绘制轮廓
    # contour_image = np.zeros_like(mask)
    # contour_image = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2RGB)
    # # 绘制轮廓
    # cv2.drawContours(contour_image, contour, -1, (0, 0, 255), 2)
    #
    # # 显示结果
    # cv2.imwrite('contour.png', contour_image)
    return contour


#########################################################
# 找到中心线
#########################################################
def getSkeleton(binary):
    binary = binary / 255
    skeleton0 = morphology.skeletonize(binary)
    skeleton = skeleton0.astype(np.uint8) * 255
    return skeleton


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
def drawCircle(temp_img, radius, centers):
    # 绘制内切圆
    # gray_img = np.asarray(gray_img)
    # qcpoint = gray_img.copy()
    # qcpoint = cv2.cvtColor(qcpoint, cv2.COLOR_GRAY2RGB)
    for i in range(len(radius)):
        cv2.circle(temp_img, centers[i], radius[i], 255, 1)


#########################################################
# 画出分割线
#########################################################
def drawSegLine(qcpoint, contour, centers, radius):
    slope = getSlope(contour, centers)

    for i, k in enumerate(slope):
        start1, end1 = calculateDelta(k, radius[i], centers[i])
        if k == float('inf'):
            start2, end2 = calculateDelta(0, radius[i], centers[i])
        elif k == 0:
            start2, end2 = calculateDelta(float('inf'), radius[i], centers[i])
        else:
            start2, end2 = calculateDelta(-1 / k, radius[i], centers[i])
        cv2.line(qcpoint, start1, end1, 255, 1)  # 替换 (255, 255, 255) 为线段的颜色，thickness 为线段的粗细
        cv2.line(qcpoint, start2, end2, 255, 1)  # 替换 (255, 255, 255) 为线段的颜色，thickness 为线段的粗细
    cv2.imwrite('temp.png', qcpoint)


#########################################################
# 求分割线斜率
#########################################################
def getSlope(contour, centers):
    slope = []
    for i, center in enumerate(centers):
        slope_pt1 = []
        slope_pt2 = []
        for point in contour:
            x, y = point[0]
            if y == center[1] - 4:
                slope_pt1.append(point[0])
            if y == center[1] + 4:
                slope_pt2.append(point[0])
        slope_pt1.sort(key=lambda item: abs(center[0] - item[0]))
        slope_pt2.sort(key=lambda item: abs(center[0] - item[0]))
        if (center[0] - slope_pt1[0][0]) * (center[0] - slope_pt2[0][0]) > 0:
            if slope_pt1[0][0] == slope_pt2[0][0]:
                slope1 = float('inf')
            else:
                slope1 = (slope_pt1[0][1] - slope_pt2[0][1]) / (slope_pt1[0][0] - slope_pt2[0][0])
        else:
            if slope_pt1[0][0] == slope_pt2[1][0]:
                slope1 = float('inf')
            else:
                slope1 = (slope_pt1[0][1] - slope_pt2[1][1]) / (slope_pt1[0][0] - slope_pt2[1][0])
        slope.append(slope1)
    return slope


#########################################################
# 求质控点分割线坐标偏移量
#########################################################
def calculateDelta(slope, length, center):
    angle = math.atan(slope)  # 计算斜角的弧度值
    # 计算水平投影长度（在 x 轴上的投影长度）
    delta_x = length if slope == 0 else length * math.cos(angle)
    # 计算垂直投影长度（在 y 轴上的投影长度）
    delta_y = length if math.isinf(slope) else length * math.sin(angle)

    x_start = int(center[0] - delta_x)
    y_start = int(center[1] - delta_y)
    x_end = int(center[0] + delta_x)
    y_end = int(center[1] + delta_y)
    start = (x_start, y_start)
    end = (x_end, y_end)

    return start, end


#########################################################
# 求平均灰度
#########################################################
def getAvgGray(temp_img, radius, centers):
    avg_gray = []
    top, bottom, left, right = [], [], [], []
    cnts = []
    for m in range(len(radius)):
        top.append(centers[m][1] - radius[m])
        bottom.append(centers[m][1] + radius[m])
        left.append(centers[m][0] - radius[m])
        right.append(centers[m][0] + radius[m])
    cnt, _ = cv2.findContours(temp_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(cnt) // 5):
        cnt1 = []
        for j in range(5 * i, 5 * i + 5):
            cnt1.append(cnt[j])
        cnts.append(cnt1)
    if len(cnt) > 5:
        cnts[0], cnts[1] = cnts[1], cnts[0]

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
    avg_gray.append(gray_avg)
    # 求四个扇形平均灰度
    # 左上
    gray_avg1 = []
    for m in range(len(cnts)):
        gray_total = 0
        count = 0
        for i in range(top[m], bottom[m]):  # 高，y轴
            for j in range(left[m], right[m]):  # 宽，x轴
                dist = cv2.pointPolygonTest(cnts[m][1], (j, i), True)
                if dist > 0:
                    gray_total += gray_img[i, j]
                    count += 1
        gray_avg1.append(gray_total // count)
    print(gray_avg1)
    avg_gray.append(gray_avg1)
    # 左下
    gray_avg2 = []
    for m in range(len(cnts)):
        gray_total = 0
        count = 0
        for i in range(top[m], bottom[m]):  # 高，y轴
            for j in range(left[m], right[m]):  # 宽，x轴
                dist = cv2.pointPolygonTest(cnts[m][2], (j, i), True)
                if dist > 0:
                    gray_total += gray_img[i, j]
                    count += 1
        gray_avg2.append(gray_total // count)
    print(gray_avg2)
    avg_gray.append(gray_avg2)
    # 右上
    gray_avg3 = []
    for m in range(len(cnts)):
        gray_total = 0
        count = 0
        for i in range(top[m], bottom[m]):  # 高，y轴
            for j in range(left[m], right[m]):  # 宽，x轴
                dist = cv2.pointPolygonTest(cnts[m][3], (j, i), True)
                if dist > 0:
                    gray_total += gray_img[i, j]
                    count += 1
        gray_avg3.append(gray_total // count)
    print(gray_avg3)
    avg_gray.append(gray_avg3)
    # 右下
    gray_avg4 = []
    for m in range(len(cnts)):
        gray_total = 0
        count = 0
        for i in range(top[m], bottom[m]):  # 高，y轴
            for j in range(left[m], right[m]):  # 宽，x轴
                dist = cv2.pointPolygonTest(cnts[m][4], (j, i), True)
                if dist > 0:
                    gray_total += gray_img[i, j]
                    count += 1
        gray_avg4.append(gray_total // count)
    print(gray_avg4)
    avg_gray.append(gray_avg4)
    return avg_gray


# mask_img:原图提取的血管区域图像
# qc:两个指控区域图像
# mask_qc:两个指控区域二值图
# skeleton:血管中心线
# contour:血管轮廓
# radius:半径
# centers:圆心
# temp_img:临时图像绘制质控点和分割线轮廓
if __name__ == '__main__':
    mask_img, qc = getQcMask()
    gray_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)[1]
    mask_qc = preprocess(qc)

    # skeleton = getSkeleton(binary_img)
    contour = getcontour()

    start = time.time()
    radius, centers = getCircle(binary_img, contour)
    end = time.time()
    print("getCircle:", (end - start), 's')

    temp_img = np.empty(gray_img.shape, dtype="uint8")
    drawCircle(temp_img, radius, centers)
    drawSegLine(temp_img, contour, centers, radius)

    avg_gray = getAvgGray(temp_img, radius, centers)

    # plt.subplot(121), plt.imshow(gray_img, cmap='gray')
    # plt.title('origin_img'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(qcpoint, cmap='gray')
    # plt.title('qcpoint_img'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # cv2.imwrite('qcpoint.png', qcpoint)
    # 绘制中心线
    # res = cv2.bitwise_xor(binary, skeleton)
