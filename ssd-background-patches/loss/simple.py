import torch

from omegaconf import DictConfig

from evaluation.detection import list_iou
from detection.detection_base import DetectionsBase


def total_loss(
    detections: DetectionsBase, ground_truthes: DetectionsBase, config: DictConfig
):
    # iou_threshold = config.iou_threshold
    iou_threshold = 0.5
    iou = list_iou(detections.xyxy, ground_truthes.xyxy)

    z = torch.where((iou >= iou_threshold), 1, 0).any(dim=1).to(dtype=torch.int)

    # loss weight
    alpha = config.tpc_weight

    tpc_score = tpc_loss(detections, z)
    fpc_score = fpc_loss(detections, z)

    return (tpc_score, fpc_score)


def tpc_loss(detections: DetectionsBase, z):
    score = -1 * (
        ((z * torch.log(1 - (detections.conf + 1e-9))).sum()) / (z.sum() + 1e-9)
    )
    return score


def fpc_loss(detections: DetectionsBase, z):
    score = -1 * (
        (((1 - z) * torch.log(detections.conf + 1e-9)).sum()) / ((1 - z).sum() + 1e-9)
    )
    return score
