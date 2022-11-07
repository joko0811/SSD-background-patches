import numpy as np
import cv2

import torch
import torch.optim as optim
from torchvision.datasets.coco import CocoDetection
from torchvision import transforms

from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS

import util
from model import yolo
from dataset import coco
from loss import total_loss
import proposed_func as pf


def get_image_from_dataset():
    train_path = "./coco2014/images/train2014/"
    train_annfile_path = "./coco2014/annotations/instances_train2014.json"

    coco_train = CocoDetection(root=train_path,
                               annFile=train_annfile_path, transform=transforms.ToTensor())
    img, target = coco_train[0:1]
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    return img


def get_image_from_file():
    image_path = "./data/dog.jpg"
    image_size = 416
    image = cv2.imread(image_path)
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(image_size)])(
            (image, np.zeros((1, 5))))[0].unsqueeze(0)
    return input_img


def save_image(orig_img, detections):
    pil_img = transforms.functional.to_pil_image(orig_img[0])
    datasets_class_names_path = "./coco2014/coco.names"
    class_names = coco.load_class_names(datasets_class_names_path)

    ann_img = util.img.draw_boxes(pil_img, detections.xyxy,
                                  detections.class_labels, detections.confidences, class_names)
    ann_img.show()


def run():
    img = get_image_from_file()
    test_loss(img)


def test_loss(orig_img):

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
        gt_nearest_idx = util.box.find_nearest_box(
            detections.xywh, ground_truthes.xywh)
        gt_box_nearest_dt = ground_truthes.xyxy[gt_nearest_idx]
        # detectionと、各detectionに一番近いground truthのiouスコアを算出
        dt_gt_iou_scores = util.box.iou(detections.xyxy, gt_box_nearest_dt)
        # dtのスコアから、gt_nearest_idxで指定されるground truthの属するクラスのみを抽出
        dt_scores_for_nearest_gt_label = detections.class_scores.gather(
            1, ground_truthes.class_labels[gt_nearest_idx, None]).squeeze()
        # 論文で提案された変数zを計算
        z = pf.calc_z(dt_scores_for_nearest_gt_label, dt_gt_iou_scores)

        # 検出と一番近い背景パッチ
        bp_nearest_idx = util.box.find_nearest_box(
            detections.xywh, background_patch_box)
        # detectionと、各detectionに一番近いground truthのiouスコアを算出
        bp_box_nearest_dt = background_patch_box[bp_nearest_idx]
        dt_bp_iou_scores = util.box.iou(detections.xyxy, bp_box_nearest_dt)
        # 論文で提案された変数rを計算
        r = pf.calc_r(dt_bp_iou_scores, detections.xyxy, ground_truthes.xyxy)

        # 損失計算用の情報を積む
        detections.set_loss_info(gt_nearest_idx, z, r)

        loss = total_loss(detections, ground_truthes)

        t = t+1
    print("success!")


def main():
    run()


if __name__ == '__main__':
    main()
