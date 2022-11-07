import torch
import math
import numpy as np

from model.yolo import detections_nms_out, detections_loss

import util


def tpc(detections: detections_loss, ground_truthes: detections_nms_out) -> float:
    """True Positive Class Loss
    """
    tpc_score = 0.

    gt_labels_for_nearest_dt = ground_truthes.class_labels[detections.nearest_gt_idx]
    calculated_class_score = detections.class_scores.detach()

    for i in range(calculated_class_score.shape[0]):
        ignore_idx = gt_labels_for_nearest_dt[i]
        calculated_class_score[i, ignore_idx] = 0

        tpc_score += (detections.z[i] *
                      torch.log(torch.max(calculated_class_score[i]))).item()

    tpc_score *= -1
    return tpc_score


def tps(detections: detections_loss, ground_truthes: detections_nms_out) -> float:
    """True Positive Shape Loss
    """
    tps_score = 0.

    # (x-x)^2+(y-y)^2+(w-w)^2+(h-h)
    dist = (
        (detections.xywh-ground_truthes.xywh[detections.nearest_gt_idx])**2)

    tps_score = torch.exp(-1*torch.sum(detections.z*torch.sum(dist, dim=1)))

    return tps_score.item()


def fpc(detections: detections_loss, ground_truthes: detections_nms_out) -> float:
    """False Positive Class Loss
    """
    fpc_score = 0.

    gt_labels_for_nearest_dt = ground_truthes.class_labels[detections.nearest_gt_idx]
    calculated_class_score = detections.class_scores.detach()

    for i in range(calculated_class_score.shape[0]):
        ignore_idx = gt_labels_for_nearest_dt[i]
        calculated_class_score[i, ignore_idx] = 0

        fpc_score += (detections.r[i] *
                      torch.log(torch.max(calculated_class_score[i]))).item()
    fpc_score *= -1
    return fpc_score


def total_loss(detections: detections_loss, ground_truthes: detections_nms_out):
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
