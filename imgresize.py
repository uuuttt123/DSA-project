import os
import cv2

img_path = r'postprocessed'
for root, dirs, files in os.walk(img_path):
    for file in files:
        img = cv2.imread(os.path.join(root, file))
        print(img.shape)
        width = img.shape[0]
        height = img.shape[1]
        img_crop = img[width-1304:width, height-1334:height]
        cv2.imwrite(os.path.join(root, file), img_crop)