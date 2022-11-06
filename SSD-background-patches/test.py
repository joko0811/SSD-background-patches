import numpy as np
import cv2

import torch
import torch.optim as optim
from torchvision.datasets.coco import CocoDetection
from torchvision import transforms

from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS

from model import yolo
from util import img, box
from dataset import coco
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

    ann_img = img.draw_boxes(pil_img, detections.xyxy,
                             detections.class_labels, detections.confidences, class_names)
    ann_img.show()


def run():
    img = get_image_from_file()
    test_loss(img)


def test_loss(orig_img):

    epoch = 1  # T in paper
    t = 0  # t in paper (iterator)

    if torch.cuda.is_available():
        gpu_img = orig_img.to(
            device='cuda:0', dtype=torch.float)
    optimizer = optim.Adam([gpu_img])

    # 素の画像を物体検出器にかけた時の出力をground truthとする
    ground_truthes = yolo.detect(gpu_img)
    ground_truthes = yolo.nms(ground_truthes)
    ground_truthes = yolo.detections_nms_out(ground_truthes[0])

    while t < epoch:
        optimizer.zero_grad()

        # t回目のパッチ適用画像から物体検出する
        detections = yolo.detect(gpu_img)
        detections = yolo.detections_loss(detections[0])

        # 検出と一番近いground truth
        gt_nearest_idx = box.find_nearest_box(
            detections.xywh, ground_truthes.xywh)

        # detectionと、各detectionに一番近いground truthのiouスコアを算出
        gt_box_nearest_dt = ground_truthes.xyxy[gt_nearest_idx]
        dt_gt_iou_scores = box.iou(detections.xyxy, gt_box_nearest_dt)

        # dtのスコアから、gt_nearest_idxで指定されるground truthの属するクラスのみを抽出
        dt_scores_for_nearest_gt_label = detections.class_scores.gather(
            1, ground_truthes.class_labels[gt_nearest_idx, None]).squeeze()
        # 論文で提案された変数zを計算
        z = pf.calc_z(dt_scores_for_nearest_gt_label, dt_gt_iou_scores)

    print("success!")


def main():
    run()


if __name__ == '__main__':
    main()
