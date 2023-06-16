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

    # 検出毎に正しい顔領域とのIOUをとり、閾値以上かそうでないかで二値化する
    z = torch.where((iou >= iou_threshold), 1, 0)

    # 検出毎に計算したIOUが閾値以上である正しい顔領域の数/正しい顔領域の合計数
    w = z.sum(dim=1) / len(ground_truthes)

    tpc_score = tpc_loss(detections, w) * config.tpc_weight
    fpc_score = fpc_loss(detections, w) * config.fpc_weight

    return (tpc_score, fpc_score)


def tpc_loss(detections: DetectionsBase, det_weight):
    score = -1 * (
        (det_weight * torch.log((1 - detections.conf) + 1e-5)).sum() / len(detections)
    )
    return score


def fpc_loss(detections: DetectionsBase, det_weight):
    score = -1 * (
        ((1 - det_weight) * torch.log(detections.conf + 1e-5)).sum() / len(detections)
    )
    return score
