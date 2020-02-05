### detect the road defect
### Zuheng Ming @ 2020.01.28

from mmdet.apis import init_detector, inference_detector, show_result
import mmcv



def main():


    config_file = '../configs/faster_rcnn_r50_fpn_1x.py'
    checkpoint_file = '../checkpoints/fast_mask_rcnn_r50_fpn_1x_20181010-e030a38f.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # test a single image and show the results
    img = '../data/coco/val2017/000000000139.jpg'#'/data/zming/datasets/RoadDamageDataset2018/Chiba/JPEGImages/Chiba_20170925172213.jpg'#'test.jpg'  # or img = mmcv.imread(img), which will only load it once
    result = inference_detector(model, img)
    # visualize the results in a new window
    show_result(img, result, model.CLASSES)
    # or save the visualization results to image files
    show_result(img, result, model.CLASSES, out_file='result.jpg')



if __name__ == '__main__':
    main()
