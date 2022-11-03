import torch
import math
import numpy as np

import util


def tpc(ditections, ground_truthes) -> float:
    """True Positive Class Loss
    """
    # 検出から本来の正解ラベルのクラススコアを抽出する
    nearest_gt_label = [ground_truthes.class_label[i]
                        for i in ditections.near_gt_idx]
    calculated_class_score = ditections.class_score.copy()
    calculated_class_score[:, nearest_gt_label] = 0

    tpc_score = -1*np.sum(
        [ditections.z[i]*np.max(calculated_class_score[i]) for i in range(ditections.total_det)])

    return tpc_score


def tps(detections, ground_truthes) -> float:
    """True Positive Shape Loss
    """
    tps_score = 0.

    d_boxes_xywh = util.box.xyxy2xywh(detections.boxes)
    g_boxes_xywh = util.box.xyxy2xywh(
        [ground_truthes.boxes[i] for i in detections.nearest_gt_idx])

    dist = ((d_boxes_xywh-g_boxes_xywh)**2)  # (x-x)^2+(y-y)^2+(w-w)^2+(h-h)
    tps_score = np.exp(-1*np.sum([detections.z*np.sum(d) for d in dist]))

    return tps_score


def fpc(ditections, ground_truthes) -> float:
    """False Positive Class Loss
    """

    # 対応するGround Truthで検出したクラスをインデックスとし、Detectionsのスコアを抽出する
    # 検出から本来の正解ラベルのクラススコアを抽出する
    nearest_gt_label = [ground_truthes.class_label[i]
                        for i in ditections.near_gt_idx]
    calculated_class_score = ditections.class_score.copy()
    calculated_class_score[:, nearest_gt_label] = 0

    fpc_score = -1*np.sum(
        [ditections.r[i]*np.max(calculated_class_score[i]) for i in range(ditections.total_det)])

    return fpc_score


def total_loss(detections, ground_truthes):
    """Returns the total loss
    Args:
      detections:
        yolo detections
        (x1, y1, x2, y2, conf, cls)
      ground_truthes:
        ground truth detections
        (x1, y1, x2, y2, conf, cls)
    """
    tpc_score = tpc(detections, ground_truthes)
    tps_score = tps(detections, ground_truthes)
    fpc_score = fpc(detections, ground_truthes)

    loss = tpc_score+tps_score+fpc_score
    return loss
