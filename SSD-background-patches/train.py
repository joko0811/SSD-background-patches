import torch
import cv2

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

    ground_truthes = yolo.detect(orig_img)
    ground_truthes = yolo.nms(ground_truthes)
    ground_truthes = yolo.ditections_base(ground_truthes)

    psnr_threshold = 0

    if torch.cuda.is_available():
        dtype = torch.float
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        gpu_img = torch.tensor(
            orig_img, device=device, dtype=dtype, requires_grad=True)

    while t < epoch:
        gpu_img.grad.zero_()

        detections = yolo.detection(gpu_img)
        detections = yolo.detections_loss(detections)

        # 検出と一番近いGround Truth
        gt_nearest_idx = box.find_nearest_box(
            detections.boxes, ground_truthes.boxes)
        # 検出と一番近い背景パッチ
        bp_nearest_idx = box.find_nearest_box(
            detections.boxes, background_patch_boxes)

        # 論文の変数zを算出する
        gt_box_nearest_dt = [ground_truthes.boxes[i] for i in gt_nearest_idx]
        dt_gt_iou_scores = box.iou(detections.boxes, gt_box_nearest_dt)
        z = pf.calc_z(
            detections.class_scores[gt_nearest_idx, ground_truthes.class_label], dt_gt_iou_scores)

        # 論文の変数rを算出する
        bp_box_nearest_dt = [background_patch_boxes[i] for i in bp_nearest_idx]
        dt_bp_iou_scores = box.iou(detections.boxes, bp_box_nearest_dt)
        r = pf.calc_r(dt_bp_iou_scores, detections.boxes, ground_truthes.boxes)

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

    patch_num = 3  # n_b in paper
    for _ in range(patch_num):
        background_patch_generation(img)


if __name__ == "__main__":
    main()
