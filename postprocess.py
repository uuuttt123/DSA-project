import cv2
import os
import numpy as np

imaPath = 'exp2/'
output = 'exp2output/'
#
# import cv2
imaList = os.listdir(imaPath)
for files in imaList:
    img = os.path.join(imaPath, files)
    path_processed = os.path.join(output, files)

    img = cv2.imread(img)
    gray = img[:, :, 0]
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大区域并填充
    area = []

    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))

    max_idx = np.argmax(area)
    max_area = cv2.contourArea(contours[max_idx])
    # print(max_area)
    for k in range(len(contours)):
        if k != max_idx:
            print(k, 1)
            cv2.fillPoly(img, [contours[k]], (0, 0, 0))

    # cv2.drawContours(img, contours[max_idx], -1, (0, 0, 255), 3)

    cv2.imwrite(path_processed, img)