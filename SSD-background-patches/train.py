import cv2

import torch
import torch.optim as optim
from torchvision.datasets.coco import CocoDetection

from util.img import pil2cv
from util import box
from model import yolo
import proposed_func as pf


def background_patch_generation(orig_img):
    """algorithm_1 in paper
    """

    epoch = 0  # T in paper
    t = 0  # t in paper (iterator)

    adv_img = orig_img.detach()  # return

    psnr_threshold = 0

    if torch.cuda.is_available():
        gpu_img = gpu_img.to(
            device='cuda:0', dtype=torch.float, requires_grad=True)
    optimizer = optim.Adam([gpu_img])

    ground_truthes = yolo.detect(gpu_img)
    ground_truthes = yolo.nms(ground_truthes)
    ground_truthes = yolo.detections_nms_out(ground_truthes[0])

    while t < epoch:
        optimizer.zero_grad()

        detections = yolo.detect(gpu_img)
        detections = yolo.detections_loss(detections[0])

        # 検出と一番近いGround Truth
        gt_nearest_idx = box.find_nearest_box(
            detections.xywh, ground_truthes.xywh)
        # 検出と一番近い背景パッチ
        bp_nearest_idx = box.find_nearest_box(
            detections.xywh, background_patch_boxes)

        # 論文の変数zを算出する
        gt_box_nearest_dt = [ground_truthes.xyxy[i] for i in gt_nearest_idx]
        dt_gt_iou_scores = box.iou(detections.xyxy, gt_box_nearest_dt)
        z = pf.calc_z(
            detections.class_scores[gt_nearest_idx, ground_truthes.class_labels], dt_gt_iou_scores)

        # 論文の変数rを算出する
        bp_box_nearest_dt = [background_patch_boxes[i] for i in bp_nearest_idx]
        dt_bp_iou_scores = box.iou(detections.xyxy, bp_box_nearest_dt)
        r = pf.calc_r(dt_bp_iou_scores, detections.xyxy, ground_truthes.xyxy)

        # 損失計算用の情報を積む
        detections.set_loss_info(gt_nearest_idx, z, r)

        loss = loss.total_loss(detections, ground_truthes)

        loss.backword()

        grad_img = gpu_img.grad()

        if t == 0:
            background_patch_boxes = pf.initial_background_patches()
        else:
            background_patch_boxes = pf.expanded_background_patches()

        perturbated_image = pf.perturbation_in_background_patches(
            grad_img, background_patch_boxes)
        perturbated_image = pf.perturbation_normalization(perturbated_image)

        adv_img = pf.update_i_with_pixel_clipping(adv_img, perturbated_image)

        if cv2.psnr() < psnr_threshold:
            break

        t += 1  # iterator increment

    return adv_img


def main():
    train_path = "./coco2014/images/train2014/"
    train_annfile_path = "./coco2014/annotations/instances_train2014.json"
    coco_train = CocoDetection(root=train_path,
                               annFile=train_annfile_path)
    img, target = coco_train[0]
    img = pil2cv(img)

    background_patch_generation(img)


if __name__ == "__main__":
    main()
