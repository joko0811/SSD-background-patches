import numpy as np
import cv2

import time

import torch
from torchvision.datasets.coco import CocoDetection
from torchvision import transforms

from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS


from util import img
from model import yolo, yolo_util
from dataset.coco import load_class_names
from dataset.simple import DirectoryDataset


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
    class_names = load_class_names(datasets_class_names_path)

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
    coco_path = "./coco2014/images/train2014/"
    coco_annfile_path = "./coco2014/annotations/instances_train2014.json"
    images_dir_path = "./testdata/evaluate/20221201_220253/"

    coco_class_names_path = "./coco2014/coco.names"
    class_names = load_class_names(coco_class_names_path)

    model = yolo.load_model(
        "weights/yolov3.cfg",
        "weights/yolov3.weights")
    model.eval()

    yolo_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((416, 416)),
    ])

    coco_set = CocoDetection(root=coco_path,
                             annFile=coco_annfile_path, transform=yolo_transforms)
    coco_loader = torch.utils.data.DataLoader(coco_set)

    dir_set = DirectoryDataset(
        image_path=images_dir_path, transform=yolo_transforms)
    dir_loader = torch.utils.data.DataLoader(dir_set)

    for i, ((dir_image, dir_image_path), (coco_image, coco_info)) in enumerate(zip(dir_loader, coco_loader)):
        if i > 5:
            break

        if torch.cuda.is_available():
            dir_image = dir_image.to(device='cuda:0', dtype=torch.float)
            coco_image = coco_image.to(device='cuda:0', dtype=torch.float)

        dir_output = model(dir_image)
        dir_nms_out = yolo_util.nms(dir_output)
        dir_detections = yolo_util.detections_base(dir_nms_out[0])

        dir_anno_img = img.tensor2annotation_image(
            dir_image, dir_detections, class_names)
        dir_anno_img.save("data/dir_"+str(i)+".png")

        coco_output = model(coco_image)
        coco_nms_out = yolo_util.nms(coco_output)
        coco_detections = yolo_util.detections_base(coco_nms_out[0])

        coco_anno_img = img.tensor2annotation_image(
            coco_image, coco_detections, class_names)
        coco_anno_img.save("data/coco_"+str(i)+".png")


def run():
    test_dataset()
    # model = yolo_util.load_model(
    #     "weights/yolov3.cfg",
    #     "weights/yolov3.weights")
    # input_path = "./data/dog.jpg"
    # output_path = "./data/dog_anno.jpg"
    # test_detect(input_path, output_path)


def main():
    run()


if __name__ == '__main__':
    main()
