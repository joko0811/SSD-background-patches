import numpy as np
import cv2

import time

import torch
from torchvision.datasets.coco import CocoDetection
from torchvision import transforms

from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS


from util import img
from model import yolo_util
from dataset import coco


def get_image_from_dataset():
    train_path = "./coco2014/images/train2014/"
    train_annfile_path = "./coco2014/annotations/instances_train2014.json"

    coco_train = CocoDetection(root=train_path,
                               annFile=train_annfile_path, transform=transforms.ToTensor())
    img, _ = coco_train[0:1]
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    return img


def get_image_from_file(image_path):
    # image_path = "./testdata/adv_image.png"
    image_size = 416
    image = cv2.imread(image_path)
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(image_size)])(
            (image, np.zeros((1, 5))))[0].unsqueeze(0)
    return input_img


def make_annotation_image(orig_img, detections):
    pil_img = transforms.functional.to_pil_image(orig_img[0])
    datasets_class_names_path = "./coco2014/coco.names"
    class_names = coco.load_class_names(datasets_class_names_path)

    ann_img = img.draw_annotations(pil_img, detections.xyxy,
                                   detections.class_labels, detections.confidences, class_names)
    return ann_img


def make_box_image(image, boxes):
    pil_img = transforms.functional.to_pil_image(image[0])
    tmp_img = img.draw_boxes(pil_img, boxes)
    return tmp_img


def save_image(image, image_path):
    pil_img = transforms.functional.to_pil_image(image[0])
    pil_img.save(image_path)


def test_detect(input_path, output_path):
    image = get_image_from_file(input_path)
    if torch.cuda.is_available():
        gpu_image = image.to(
            device='cuda:0', dtype=torch.float)
    yolo_out = yolo_util.detect(gpu_image)
    nms_out = yolo_util.nms(yolo_out)
    detections = yolo_util.detections_base(nms_out[0])
    anno_image = make_annotation_image(image, detections)
    anno_image.save(output_path)


def test_dataset():
    train_path = "./coco2014/images/train2014/"
    train_annfile_path = "./coco2014/annotations/instances_train2014.json"

    coco_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((416, 416)),
    ])

    train_set = CocoDetection(root=train_path,
                              annFile=train_annfile_path, transform=coco_transforms)

    train_loader = torch.utils.data.DataLoader(train_set)
    for i, (image, info) in enumerate(train_loader):
        print(i)


def run():
    model = yolo_util.load_model(
        "weights/yolov3.cfg",
        "weights/yolov3.weights")
    test_dataset()
    # input_path = "./testdata/monitor/20221127_171928/adv_image.png"
    # output_path = "./testdata/monitor/20221127_171928/anno_image.png"
    # test_detect(input_path, output_path)


def main():
    run()


if __name__ == '__main__':
    main()
