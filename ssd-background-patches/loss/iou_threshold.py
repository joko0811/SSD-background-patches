import torch

from omegaconf import DictConfig

from evaluation.detection import list_iou
from detection.detection_base import DetectionsBase
from loss.objdet_base import ObjectDetectionBaseLoss


class SimpleTPCLoss(ObjectDetectionBaseLoss):

    def __call__(self, detections: DetectionsBase, ground_truthes: DetectionsBase):
        z = calc_det_weight(
            detections, ground_truthes, iou_threshold=self.iou_threshold
        )

        score = -1 * (
            ((z * torch.log((1 - detections.conf) + 1e-9)).sum()) / (z.sum() + 1e-9)
        )
        return score


class SimpleFPCLoss(ObjectDetectionBaseLoss):
    def __call__(self, detections: DetectionsBase, ground_truthes: DetectionsBase):
        z = calc_det_weight(
            detections, ground_truthes, iou_threshold=self.iou_threshold
        )

        score = -1 * (
            (((1 - z) * torch.log(detections.conf + 1e-9)).sum())
            / ((1 - z).sum() + 1e-9)
        )
        return score


def calc_det_weight(
    detections: DetectionsBase, ground_truthes: DetectionsBase, iou_threshold=0.5
):
    iou = list_iou(detections.xyxy, ground_truthes.xyxy)
    z = torch.where((iou >= iou_threshold), 1, 0).any(dim=1).to(dtype=torch.int)
    return z
