from evaluation.detection import list_iou
import torch


def total_loss(detections, ground_truthes, config):
    iou_score = _iou(detections=detections, ground_truthes=ground_truthes)
    return iou_score.mean()


def _iou(detections, ground_truthes):
    gt_dt_iou = list_iou(ground_truthes.xyxy, detections.xyxy)
    dt_iou_sum = gt_dt_iou.sum(axis=0)

    z = (gt_dt_iou != 0).to(dtype=torch.float).sum(axis=0)

    normalize_dt_iou_sum = dt_iou_sum / z

    return normalize_dt_iou_sum
