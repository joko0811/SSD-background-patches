import torch
import numpy as np

from torchvision import transforms
from torchvision.datasets.coco import CocoDetection

from pytorchyolo import models
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS

from util.img import pil2cv
from util.detection import nms


def initial_background_patches():
    return


def expanded_background_patches():
    return


def perturbation_in_background_patches():
    return


def pertrbation_normalization():
    return


def update_i_with_pixel_clipping():
    return


def detect(img):

    # image setup
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(416)])(
            (img, np.zeros((1, 5))))[0].unsqueeze(0)
    if torch.cuda.is_available():
        input_img = input_img.to("cuda")

    model = models.load_model(
        "weights/yolov3.cfg",
        "weights/yolov3.weights")

    model.eval()  # Set model to evaluation mode

    # Get detections
    with torch.no_grad():
        # yolo_out=[center_x,center_y,w,h,confidence_score,class_scores...]
        yolo_out = model(input_img)

        # yolo_out=[left_x,top_y,right_x,bottom_y,class_scores...]
        return yolo_out


def background_patch_generation():
    train_path = "./coco2014/images/train2014/"
    train_annfile_path = "./coco2014/annotations/instances_train2014.json"

    epoch = 0  # t in paper
    patch_num = 3  # n_b in paper

    coco_train = CocoDetection(root=train_path,
                               annFile=train_annfile_path)
    img, target = coco_train[0]
    img = pil2cv(img)

    yolo_out = detect(img)
    yolo_out = nms(yolo_out)


def main():
    background_patch_generation()


if __name__ == "__main__":
    main()
