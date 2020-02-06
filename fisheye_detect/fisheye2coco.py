#### Convert dataset fisheye dataset to coco format
### Zuheng Ming @ 20200130
### Licence:
########

import os
import cv2
import shutil
import glob
import json
import numpy as np
from sklearn.model_selection import KFold

data_dst = "/data/zming/GH/fisheye_detect/videos/fullannotation_3videos_clean_coco_500"
data_src = "/data/zming/GH/fisheye_detect/videos/fullannotation_3videos_clean"
nfold = 10
ifold = 0


def main():

    if not os.path.isdir(os.path.join(data_dst,"annotations")):
        os.mkdir(os.path.join(data_dst,"annotations"))
    if not os.path.isdir(os.path.join(data_dst,"train2017")):
        os.mkdir(os.path.join(data_dst,"train2017"))
    if not os.path.isdir(os.path.join(data_dst,"val2017")):
        os.mkdir(os.path.join(data_dst,"val2017"))
    if not os.path.isdir(os.path.join(data_dst,"test2017")):
        os.mkdir(os.path.join(data_dst,"test2017"))


    # # copy the images to the train2017 and val2017
    # folders = os.listdir(data_src)
    # folders.sort()
    # images_all = []
    # images_train = []
    # images_val = []
    # idx_train_all =[]
    # idx_val_all =[]
    #
    # for folder in folders:
    #     images = glob.glob(os.path.join(data_src,folder,'*.png'))
    #     images_all += images
    #
    # kf = KFold(n_splits=nfold, random_state=666, shuffle=True)
    # i = 0
    #
    # for idx_train, idx_val in kf.split(images_all):
    #     idx_train_all.append([])
    #     idx_train_all[i].append(idx_train)
    #     idx_val_all.append([])
    #     idx_val_all[i].append(idx_val)
    #     print('train:', idx_train, 'test', idx_val)
    #     i += 1
    #
    # idx_train = idx_train_all[ifold][0]
    # idx_val = idx_val_all[ifold][0]
    #
    # images_train = [images_all[x] for x in idx_train]
    # images_val = [images_all[x] for x in idx_val]
    #
    # for image in images_train:
    #     print("%s..."%image)
    #     sample = str.split(image, '/')[-2:]
    #     sample = sample[0]+'_'+ sample[1]
    #     sample = sample[:-4]
    #
    #     img = cv2.imread(image)
    #     img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    #     cv2.imwrite(os.path.join(data_dst, "train2017", sample+".jpg"), img)
    #
    #     # shutil.copyfile(image,
    #     #                 os.path.join(data_dst, "train2017", sample+".jpg"))
    # for image in images_val:
    #     print("%s..."%image)
    #     sample = str.split(image, '/')[-2:]
    #     sample = sample[0]+'_'+ sample[1]
    #     sample = sample[:-4]
    #     img = cv2.imread(image)
    #     img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    #     cv2.imwrite(os.path.join(data_dst, "val2017", sample + ".jpg"), img)
    #
    #     # shutil.copyfile(image,
    #     #                 os.path.join(data_dst, "val2017", sample+".jpg"))

    json.dump(gen_instance(data_src, os.path.join(data_dst, "train2017")),
              open(os.path.join(data_dst, "annotations", "instances_train2017.json"), "w"))
    json.dump(gen_instance(data_src, os.path.join(data_dst, "val2017")),
              open(os.path.join(data_dst, "annotations", "instances_val2017.json"), "w"))
    # json.dump(gen_instance(data_src, "train"), open(os.path.join(data_dst,"annotations", "instances_train2017.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # json.dump(gen_instance(data_src, "val"), open(os.path.join(data_dst,"annotations", "instances_val2017.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def gen_instance(dir, dir_imgs):
    ## convert the annotations to coco format
    dir_dst = str.split(dir_imgs, '/')[-2]
    img_size = int(str.split(dir_dst, '_')[-1])
    images = []
    annotations = []
    img_h = img_size#3000
    img_w = img_size#3000
    img_id = 0
    ann_id = 0
    imgs = glob.glob(os.path.join(dir_imgs, '*.jpg'))

    ## read the original image size
    img = imgs[0]
    img = str.split(img, '/')[-1]
    idx = len(img) - img[::-1].index('_') - 1
    img_name = img[idx + 1:]
    folder = img[:idx]
    image_original = cv2.imread(os.path.join(dir, folder, img_name[:-3] + 'png'))
    img_h_original, img_w_original, _ = image_original.shape
    resize_scale_h = img_h_original / img_h
    resize_scale_w = img_w_original / img_w
    # img_resize = cv2.resize(image_original, (img_size, img_size), interpolation=cv2.INTER_AREA)

    for img in imgs:
        print("%s" % img)
        # im = cv2.imread(img)
        img = str.split(img, '/')[-1]
        idx = len(img)-img[::-1].index('_')-1
        img_name = img[idx+1:]
        label = img_name[:-3]+'txt'
        folder = img[:idx]
        with open(os.path.join(dir, folder, label), "r") as f:
            image = {}
            image["height"] = img_h
            image["width"] = img_w
            image["id"] = img_id
            image["file_name"] = img
            images.append(image)

            for line in f.readlines():
                [xmin, ymin, x_len, y_len, label] = str.split(line, " ")
                xmin = float(xmin)/resize_scale_w
                ymin = float(ymin)/resize_scale_h
                x_len = float(x_len)/resize_scale_w
                y_len = float(y_len)/resize_scale_h

                label = int(label)
                #print('%d'%label)
                if label < 0 or label >0:
                    print('%s'%img_name)

                min_x = max(0.0, xmin)
                min_y = max(0.0, ymin)
                max_x = min(xmin+x_len, img_w)
                max_y = min(ymin+y_len, img_h)
                area = int(x_len*y_len)
                ## draw the bbox ground truth
                # cv2.rectangle(im, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)



                annotation = {}
                annotation["id"] = ann_id
                annotation["image_id"] = img_id
                annotation["category_id"] = int(label)
                annotation["segmentation"] = [[min_x,min_y, min_x,min_y+0.5*img_h, min_x,max_y, min_x+0.5*img_w,max_y, max_x,max_y, max_x,max_y-0.5*img_h, max_x,min_y, max_x-0.5*img_w,min_y]]
                annotation["bbox"] = [min_x,min_y,x_len,y_len]
                annotation["iscrowd"] = 0
                annotation["area"] = area
                annotations.append(annotation)

                ann_id += 1

            img_id += 1

            # cv2.imshow('im', im)
            # cv2.waitKey(0)

    instance = {}
    instance["info"] = "fisheye head detect dataset Challenge 2018"

    licence = {}
    licence["url"] =  "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    licence["id"] =  1
    licence["name"] =  "Attribution-NonCommercial-ShareAlike License"
    instance["licence"] = licence
    instance["images"] = images
    instance["annotations"] = annotations

    categories = []
    category = {}
    category["supercategory"] = "head"
    category["id"] = 0
    category["name"] = "head"
    categories.append(category)

    instance["categories"] = categories

    return instance

if __name__ == "__main__":
    main()