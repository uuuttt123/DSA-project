import cv2
import numpy as np
from skimage import morphology

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
    cv2.rectangle(qcmask, (rect[0]['left'], rect[0]['top']), (rect[0]['right'], rect[0]['bottom']), 255, -1)
    cv2.rectangle(qcmask, (rect[1]['left'], rect[1]['top']), (rect[1]['right'], rect[1]['bottom']), 255, -1)
    # qc_img = cv2.bitwise_and(mask_img, mask_img, mask=qcmask)   #原图上的掩膜
    # qc_img = cv2.cvtColor(qc_img, cv2.COLOR_BGR2GRAY)
    # 质控区图像列表
    qc = [mask_img[rect[0]['top']:rect[0]['bottom'], rect[0]['left']:rect[0]['right']],
          mask_img[rect[1]['top']:rect[1]['bottom'], rect[1]['left']:rect[1]['right']]]
    # 读取图片，转灰度
    # qc = qc[:, :, 0]
    qc = [cv2.cvtColor(qc[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(qc[1], cv2.COLOR_BGR2GRAY)]
    # cv2.namedWindow('maskedimg', 0)
    # cv2.imshow('maskedimg', qc_img)
    # cv2.waitKey(0)
    return mask_img, qc


#########################################################
# 质控点轮廓内切圆
#########################################################
def preprocess(qc):
    # mask_qcimg = cv2.threshold(qc_img, 1, 255, cv2.THRESH_BINARY)[1]
    # 二值化
    mask_qc = [cv2.threshold(qc[0], 1, 255, cv2.THRESH_BINARY)[1], cv2.threshold(qc[1], 1, 255, cv2.THRESH_BINARY)[1]]
    return mask_qc


# 识别最大轮廓
def getcontour(mask_qc):
    # contours = [cv2.findContours(mask_qc[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0],
    #             cv2.findContours(mask_qc[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]]
    # img = np.empty(mask_qc[0].shape)
    # cv2.drawContours(img, contours[0][1], -1, 255)
    # cv2.namedWindow('cnt', 0)
    # cv2.imshow('cnt', img)
    # cv2.waitKey(0)
    # # 找到最大区域轮廓
    # area = []
    # max_idx = []
    # for i in range(len(contours)):
    #     for j, cnt in enumerate(contours[i]):
    #         area.append(cv2.contourArea(cnt))
    #     max_idx.append(np.argmax(area))
    # contours[0] = contours[0][max_idx[0]]
    # contours[1] = contours[1][max_idx[1]]
    # print(max_idx)
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
    # radius = []
    # centers = []
    # for i, mask in enumerate(mask_qc):
    #     maxdist = 0
    #     pt = (0, 0)
    #     for j in range(mask.shape[1]):
    #         dist = cv2.pointPolygonTest(contours[i], (j, y_qc[i]), True)
    #         if dist > maxdist:
    #             maxdist = dist
    #             pt = (j, y_qc[i])
    #     radius.append(int(maxdist))
    #     centers.append(pt)
    # cv2.circle(mask_qc[0], center, radius, (0, 0, 255), -1)
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
    # qc_mask = np.zeros(gray_img.shape, dtype="uint8")
    # 绘制内切圆
    cv2.circle(gray_img, centers[0], radius[0], 255, -1)
    cv2.circle(gray_img, centers[1], radius[1], 255, -1)
    # 绘制圆心
    # cv2.circle(result, maxDistPt, 1, (0, 255, 0), 2, 1, 0)
    # qcpoint = cv2.bitwise_and(gray_img, gray_img, mask=qc_mask)
    cv2.namedWindow('qcpoint', 0)
    cv2.imshow('qcpoint', gray_img)
    cv2.waitKey(0)
    return gray_img


#########################################################
# 求平均灰度
#########################################################
def getAvgGray(gray_img, contour, radius, centers):
    top = [centers[0][1] - radius[0], centers[1][1] - radius[1]]
    bottom = [centers[0][1] + radius[0], centers[1][1] + radius[1]]
    left = [centers[0][0] - radius[0], centers[1][0] - radius[1]]
    right = [centers[0][0] + radius[0], centers[1][0] + radius[1]]
    circle1 = np.empty(gray_img.shape, dtype="uint8")
    cv2.circle(circle1, centers[0], radius[0], 255, -1)
    cnt1 = cv2.findContours(circle1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    circle2 = np.empty(gray_img.shape, dtype="uint8")
    cv2.circle(circle2, centers[1], radius[1], 255, -1)
    cnt2 = cv2.findContours(circle2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = [cnt1, cnt2]
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
    # gray_total = 0
    # count = 0
    # for i in range(qcpoint.shape[0]):
    #     for j in range(qcpoint.shape[1]):
    #         dist = cv2.pointPolygonTest(contours[0], (j, i), True)
    #         if dist > 0 and j < maxDistPt[0] and i < maxDistPt[1]:
    #             gray_total += qcpoint[j, i]
    #             count += 1
    # gray_avg1 = gray_total // count
    # print(gray_avg1)
    # gray_total = 0
    # count = 0
    # for i in range(qcpoint.shape[0]):
    #     for j in range(qcpoint.shape[1]):
    #         dist = cv2.pointPolygonTest(contours[0], (j, i), True)
    #         if dist > 0 and j > maxDistPt[0] and i < maxDistPt[1]:
    #             gray_total += qcpoint[j, i]
    #             count += 1
    # gray_avg2 = gray_total // count
    # print(gray_avg2)
    # gray_total = 0
    # count = 0
    # for i in range(qcpoint.shape[0]):
    #     for j in range(qcpoint.shape[1]):
    #         dist = cv2.pointPolygonTest(contours[0], (j, i), True)
    #         if dist > 0 and j < maxDistPt[0] and i > maxDistPt[1]:
    #             gray_total += qcpoint[j, i]
    #             count += 1
    # gray_avg3 = gray_total // count
    # print(gray_avg3)
    # gray_total = 0
    # count = 0
    # for i in range(qcpoint.shape[0]):
    #     for j in range(qcpoint.shape[1]):
    #         dist = cv2.pointPolygonTest(contours[0], (j, i), True)
    #         if dist > 0 and j > maxDistPt[0] and i > maxDistPt[1]:
    #             gray_total += qcpoint[j, i]
    #             count += 1
    # gray_avg4 = gray_total // count
    # print(gray_avg4)


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
    skeleton = getSkeleton(binary_img)
    skeletons = [skeleton[rect[0]['top']:rect[0]['bottom'], rect[0]['left']:rect[0]['right']],
                 skeleton[rect[1]['top']:rect[1]['bottom'], rect[1]['left']:rect[1]['right']]]
    contour = getcontour(mask_qc)
    radius, centers = getCircle(binary_img, contour)
    print(radius, centers)
    avg_gray = getAvgGray(gray_img, contour, radius, centers)
    qcpoint = drawCircle(gray_img, radius, centers)
    # 绘制中心线
    # res = cv2.bitwise_xor(binary, skeleton)
