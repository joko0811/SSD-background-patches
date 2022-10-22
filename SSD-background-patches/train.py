import torch
import numpy as np
import cv2

from torchvision import transforms
from torchvision.datasets.coco import CocoDetection

from pytorchyolo import models
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS

import loss
from util.img import pil2cv
from util.detection import nms


def initial_background_patches():
    return


def expanded_background_patches():
    return


def perturbation_in_background_patches():
    return


def perturbation_normalization():
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


def background_patch_generation(orig_img):
    """algorithm_1 in paper

    """

    epoch = 0  # T in paper
    t = 0  # t in paper (iterator)

    adv_img = orig_img.detach()  # return

    grand_truth = detect(orig_img)
    grand_truth = nms(grand_truth)

    psnr_threshold = 0

    while t < epoch:
        gradient = (loss.tpc()+loss.tps()+loss.fpc())
        # TODO: calc grad

        if t == 0:
            background_patch = initial_background_patches()
        else:
            yolo_out = detect(orig_img)
            yolo_out = nms(yolo_out)
            background_patch = expanded_background_patches(yolo_out)

        perturbated_image = perturbation_in_background_patches(
            gradient, background_patch)
        perturbated_image = perturbation_normalization(perturbated_image)

        adv_img = update_i_with_pixel_clipping(adv_img, perturbated_image)
        if cv2.psnr() < psnr_threshold:
            break

        t += 1

    return adv_img


def main():
    train_path = "./coco2014/images/train2014/"
    train_annfile_path = "./coco2014/annotations/instances_train2014.json"
    coco_train = CocoDetection(root=train_path,
                               annFile=train_annfile_path)
    img, target = coco_train[0]
    img = pil2cv(img)

    patch_num = 3  # n_b in paper
    for _ in range(patch_num):
        background_patch_generation(img)


if __name__ == "__main__":
    main()
