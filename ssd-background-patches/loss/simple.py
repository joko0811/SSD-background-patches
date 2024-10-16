import torch

from detection.detection_base import DetectionsBase
from loss.objdet_base import ObjectDetectionBaseLoss


class SimpleTPCLoss(ObjectDetectionBaseLoss):

    def __call__(self, detections: DetectionsBase, ground_truthes: DetectionsBase):

        score = -1 * (
            torch.log((1 - detections.conf) + 1e-9).sum()
            # / len(detections)
        )

        return score


class SimpleFPCLoss(ObjectDetectionBaseLoss):

    def __call__(self, detections: DetectionsBase, ground_truthes: DetectionsBase):

        score = -1 * (
            torch.log(detections.conf + 1e-9).sum()
            # / len(detections)
        )

        return score
