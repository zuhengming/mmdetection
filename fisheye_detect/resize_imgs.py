#### Convert dataset fisheye dataset to coco format
### Zuheng Ming @ 20200130
### Licence:
########

import os
import cv2
import shutil
import glob

def main():
    images = glob.glob('../data/coco_fisheye/val2017/*.jpg')
    for image in images:
        img = cv2.imread(image)
        img = cv2.resize(img, (500, 500), interpolation = cv2.INTER_AREA)
        cv2.imwrite(image, img)
if __name__ == '__main__':
    main()