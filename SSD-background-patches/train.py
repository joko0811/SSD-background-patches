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

    epoch = 1  # T in paper
    t = 0  # t in paper (iterator)
    psnr_threshold = 0

    if torch.cuda.is_available():
        gpu_img = orig_img.to(
            device='cuda:0', dtype=torch.float)

    with torch.no_grad():
        # 素の画像を物体検出器にかけた時の出力をground truthとする
        ground_truthes = yolo.detect(gpu_img)
        ground_truthes = yolo.nms(ground_truthes)
        ground_truthes = yolo.detections_nms_out(ground_truthes[0])

    n_b = 3  # 論文内で定められたパッチ生成枚数を指定するためのパラメータ
    background_patch_box = torch.zeros(
        [ground_truthes.total_det*n_b, 4], device=gpu_img.device)

    optimizer = optim.Adam([gpu_img])

    while t < epoch:
        # t回目のパッチ適用画像から物体検出する
        detections = yolo.detect_with_grad(gpu_img)
        detections = yolo.detections_loss(detections[0])

        # 検出と一番近いground truth
        gt_nearest_idx = box.find_nearest_box(
            detections.xywh, ground_truthes.xywh)
        gt_box_nearest_dt = ground_truthes.xyxy[gt_nearest_idx]
        # detectionと、各detectionに一番近いground truthのiouスコアを算出
        dt_gt_iou_scores = box.iou(detections.xyxy, gt_box_nearest_dt)
        # dtのスコアから、gt_nearest_idxで指定されるground truthの属するクラスのみを抽出
        dt_scores_for_nearest_gt_label = detections.class_scores.gather(
            1, ground_truthes.class_labels[gt_nearest_idx, None]).squeeze()
        # 論文で提案された変数zを計算
        z = pf.calc_z(dt_scores_for_nearest_gt_label, dt_gt_iou_scores)

        # 検出と一番近い背景パッチ
        bp_nearest_idx = box.find_nearest_box(
            detections.xywh, background_patch_box)
        # detectionと、各detectionに一番近いground truthのiouスコアを算出
        bp_box_nearest_dt = background_patch_box[bp_nearest_idx]
        dt_bp_iou_scores = box.iou(detections.xyxy, bp_box_nearest_dt)
        # 論文で提案された変数rを計算
        r = pf.calc_r(dt_bp_iou_scores, detections.xyxy, ground_truthes.xyxy)

        # 損失計算用の情報を積む
        detections.set_loss_info(gt_nearest_idx, z, r)

        loss = loss.total_loss(detections, ground_truthes)

        optimizer.zero_grad()
        loss.backward()

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
