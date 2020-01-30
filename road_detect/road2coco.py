#### Convert dataset RoadDamagaDataset2018 to coco format
### Zuheng Ming @ 20200130
### Licence:
########

import os
import cv2
import shutil
import glob
import json
import numpy as np

data_dst = "/data/zming/datasets/RoadDamageDataset2018_coco"
data_src = "/data/zming/datasets/RoadDamageDataset2018"

def main():

    if not os.path.isdir(os.path.join(data_dst,"annotations")):
        os.mkdir(os.path.join(data_dst,"annotations"))
    if not os.path.isdir(os.path.join(data_dst,"train2017")):
        os.mkdir(os.path.join(data_dst,"train2017"))
    if not os.path.isdir(os.path.join(data_dst,"val2017")):
        os.mkdir(os.path.join(data_dst,"val2017"))
    if not os.path.isdir(os.path.join(data_dst,"test2017")):
        os.mkdir(os.path.join(data_dst,"test2017"))

    json.dump(gen_instance(data_src, "train"), open(os.path.join(data_dst,"annotations", "instances_train2017.json"), "w"))
    json.dump(gen_instance(data_src, "val"), open(os.path.join(data_dst,"annotations", "instances_val2017.json"), "w"))
    #json.dump(gen_instance(data_src, "train"), open(os.path.join(data_dst,"annotations", "instances_train2017.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    #json.dump(gen_instance(data_src, "val"), open(os.path.join(data_dst,"annotations", "instances_val2017.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    ## copy the images to the train2017 and val2017
    # folders = os.listdir(data_src)
    # folders.sort()
    # for folder in folders:
    #     print("%s..."%folder)
    #     with open(os.path.join(data_src,folder,"ImageSets", "Main", "train.txt"), "r") as f:
    #         samples = f.readlines()
    #         for sample in samples:
    #             sample = sample[: -1]
    #             shutil.copyfile(os.path.join(data_src, folder, "JPEGImages", sample+".jpg"),
    #                             os.path.join(data_dst, "train2017", sample+".jpg"))
    #     with open(os.path.join(data_src,folder,"ImageSets", "Main", "val.txt"), "r") as f:
    #         samples = f.readlines()
    #         for sample in samples:
    #             sample = sample[: -1]
    #             shutil.copyfile(os.path.join(data_src, folder, "JPEGImages", sample + ".jpg"),
    #                             os.path.join(data_dst, "val2017", sample + ".jpg"))


def gen_instance(dir, type):
    ## convert the annotations to coco format
    images = []
    annotations = []
    img_h = 600
    img_w = 600
    img_id = 0
    ann_id = 0
    folders = os.listdir(dir)
    folders.sort()
    for folder in folders:
        print("%s..." % folder)
        with open(os.path.join(dir,folder,"ImageSets", "Main", "%s.txt"%type), "r") as f:
            samples = f.readlines()
        files = glob.glob(os.path.join(dir, folder, "labels", "%s_*.txt" % folder))
        for file in files:
            file_name = str.split(file,"/")[-1]
            img_name = str.split(file_name,".")[0]+"\n"
            if img_name == 'Numazu_20170906095338\n':
                print('%s'%img_name)
            if img_name not in samples:
                continue
            with open(file, "r") as f:
                image = {}
                image["height"] = img_h
                image["width"] = img_w
                image["id"] = img_id
                image["file_name"] = file_name.replace(".txt", ".jpg")
                images.append(image)
                for line in f.readlines():
                    [label, xcenter, ycenter, x_len, y_len] = str.split(line, " ")
                    label = int(label)
                    if label < 0 or label >7:
                        print('%s'%file_name)
                    xcenter = float(xcenter) * img_w
                    ycenter = float(ycenter) * img_h
                    x_len = float(x_len) * img_w
                    y_len = float(y_len) * img_h

                    ymin = ycenter - y_len / 2.0
                    ymax = ycenter + y_len / 2.0
                    xmin = xcenter - x_len / 2.0
                    xmax = xcenter + x_len / 2.0

                    min_x = max(0.0, xmin)
                    min_y = max(0.0, ymin)
                    max_x = min(xmax, img_w)
                    max_y = min(ymax, img_h)

                    annotation = {}
                    annotation["id"] = ann_id
                    annotation["image_id"] = img_id
                    annotation["category_id"] = int(label)
                    annotation["segmentation"] = [[min_x,min_y, min_x,min_y+0.5*img_h, min_x,max_y, min_x+0.5*img_w,max_y, max_x,max_y, max_x,max_y-0.5*img_h, max_x,min_y, max_x-0.5*img_w,min_y]]
                    annotation["bbox"] = [min_x,min_y,x_len,y_len]
                    annotation["iscrowd"] = 0
                    annotation["area"] = 1.0
                    annotations.append(annotation)

                    ann_id += 1

            img_id += 1

    instance = {}
    instance["info"] = "road damage dataset Challenge 2018"

    licence = {}
    licence["url"] =  "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    licence["id"] =  1
    licence["name"] =  "Attribution-NonCommercial-ShareAlike License"
    instance["licence"] = licence
    instance["images"] = images
    instance["annotations"] = annotations

    categories = []
    category = {}
    category["supercategory"] = "linearcrack"
    category["id"] = 0
    category["name"] = "D00"
    categories.append(category)
    category = {}
    category["supercategory"] = "linearcrack"
    category["id"] = 1
    category["name"] = "D01"
    categories.append(category)
    category = {}
    category["supercategory"] = "linearcrack"
    category["id"] = 2
    category["name"] = "D10"
    categories.append(category)
    category = {}
    category["supercategory"] = "linearcrack"
    category["id"] = 3
    category["name"] = "D11"
    categories.append(category)
    category = {}
    category["supercategory"] = "alligatorcrack"
    category["id"] = 4
    category["name"] = "D20"
    categories.append(category)
    category = {}
    category["supercategory"] = "corruption"
    category["id"] = 5
    category["name"] = "D40"
    categories.append(category)
    category = {}
    category["supercategory"] = "corruption"
    category["id"] = 6
    category["name"] = "D43"
    categories.append(category)
    category = {}
    category["supercategory"] = "corruption"
    category["id"] = 7
    category["name"] = "D44"
    categories.append(category)

    instance["categories"] = categories

    return instance

if __name__ == "__main__":
    main()