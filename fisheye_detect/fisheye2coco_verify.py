#### Convert dataset fisheye dataset to coco format
### Zuheng Ming @ 20200130
### Licence:
########

import os
import cv2
import glob
import json


data_dst = "/data/zming/GH/fisheye_detect/videos/fullannotation_3videos_clean_coco_1000"
folder = 'train2017'

def main():
    images = glob.glob(os.path.join(data_dst, folder, '*.jpg'))
    with open(os.path.join(data_dst, 'annotations', 'instances_%s.json'%folder)) as f:
        data = json.load(f)
        for image in images:
            #bboxs = []
            img = cv2.imread(image)
            # cv2.imshow('img', img)
            # cv2.waitKey()
            image_name = str.split(image, '/')[-1]
            for p in data['images']:
                if p['file_name'] == image_name:
                    im_id = p['id']
            for p in data['annotations']:
                if p['image_id'] == im_id:
                    bbox = p['bbox']
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0,255,0), 2)
            cv2.imshow('img', img)
            cv2.waitKey()

    return


if __name__ == "__main__":
    main()