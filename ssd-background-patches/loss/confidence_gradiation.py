from enum import IntEnum

import torch

from evaluation.detection import list_iou
from detection.detection_base import DetectionsBase
from loss.objdet_base import ObjectDetectionBaseLoss


class ConfidenceGradiationTPCLoss(ObjectDetectionBaseLoss):

    def __init__(self, iou_threshold_gradiation=[0.3, 0.6]):
        self.iou_threshold_gradiation = iou_threshold_gradiation

    def __call__(self, detections: DetectionsBase, ground_truthes: DetectionsBase):
        class_label = gradient_classification(
            detections,
            ground_truthes,
            iou_threshold_gradiation=self.iou_threshold_gradiation,
        )
        w = (class_label == len(self.iou_threshold_gradiation)).to(dtype=torch.int)

        score = -1 * (
            (w * torch.log((1 - detections.conf) + 1e-9)).sum()
            # / len(detections)
        )

        return score


class ConfidenceGradiationFPCLoss(ObjectDetectionBaseLoss):

    def __init__(self, iou_threshold_gradiation=[0.3, 0.6]):
        self.iou_threshold_gradiation = iou_threshold_gradiation

    def __call__(self, detections: DetectionsBase, ground_truthes: DetectionsBase):
        class_label = gradient_classification(
            detections,
            ground_truthes,
            iou_threshold_gradiation=self.iou_threshold_gradiation,
        )
        # w = (class_label == (class_label.sort()[-2])).to(dtype=torch.int)
        w = (class_label == (len(self.iou_threshold_gradiation) - 1)).to(
            dtype=torch.int
        )

        score = -1 * (
            (w * torch.log(detections.conf + 1e-9)).sum()
            # / len(detections)
        )

        return score


def gradient_classification(
    detections: DetectionsBase,
    ground_truthes: DetectionsBase,
    iou_threshold_gradiation=[0.3, 0.6],
):
    iou = list_iou(detections.xyxy, ground_truthes.xyxy).max(dim=1).values

    # iouの各値を昇順ソート済みのiou_threshold_gradiationに照準ソートを崩さずに挿入した時iou_threshold_gradiationのどのインデックスに入るかを求める
    # 同値の場合はインデックスの前、例えばiou_threshold_gradiation=[0.3, 0.6]でiou=0.3の時は0
    class_label = torch.searchsorted(
        torch.tensor(iou_threshold_gradiation).to(device=iou.device), iou
    )

    return class_label


class BorderlineMode(IntEnum):
    INSIDE = 0
    ON_LINE = 1
    OUTSIDE = 2


class BorderlineLossBase(ObjectDetectionBaseLoss):
    def __init__(
        self,
        theta_F: float = 0.3,
        theta_T: float = 0.6,
        mode: BorderlineMode = BorderlineMode.ON_LINE,
        normalize: bool = False,
    ):
        self.iou_threshold_gradiation = [theta_F, theta_T]
        self.mode = mode
        self.normalize = normalize

    def _calc_borderline_judgement_var(
        self, detections: DetectionsBase, ground_truthes: DetectionsBase
    ):
        class_label = gradient_classification(
            detections,
            ground_truthes,
            iou_threshold_gradiation=self.iou_threshold_gradiation,
        )
        b = (class_label == int(self.mode)).to(dtype=torch.int)
        return b


class BorderlineTPCLoss(BorderlineLossBase):

    def __call__(self, detections: DetectionsBase, ground_truthes: DetectionsBase):
        b = self._calc_borderline_judgement_var(detections, ground_truthes)

        score = -1 * ((b * torch.log((1 - detections.conf) + 1e-9)).sum())

        if self.normalize and len(b.nonzero()) > 0:
            score = score / len(b.nonzero())
        return score


class BorderlineFPCLoss(BorderlineLossBase):

    def __call__(self, detections: DetectionsBase, ground_truthes: DetectionsBase):
        b = self._calc_borderline_judgement_var(detections, ground_truthes)

        score = -1 * ((b * torch.log(detections.conf + 1e-9)).sum())

        if self.normalize and len(b.nonzero()) > 0:
            score = score / len(b.nonzero())
        return score


class BorderlineNumberLoss(BorderlineLossBase):
    """
    The intention of the design is to increase the number of inference results for which the borderline judgment is true. For this reason, the loss value decreases as the number of inference results for which the borderline judgment is true increases.
    The "normalize" argument is not used.
    """

    def __call__(self, detections: DetectionsBase, ground_truthes: DetectionsBase):
        b = self._calc_borderline_judgement_var(detections, ground_truthes)

        score = torch.exp(-b.sum())

        return score
