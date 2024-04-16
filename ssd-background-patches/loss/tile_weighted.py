import torch

from omegaconf import DictConfig

from evaluation.detection import list_iou
from detection.detection_base import DetectionsBase
from loss.objdet_base import ObjectDetectionBaseLoss


class TileWeightedTPCLoss(ObjectDetectionBaseLoss):

    def __call__(self, detections: DetectionsBase, ground_truthes: DetectionsBase):
        w = calc_det_weight(
            detections, ground_truthes, iou_threshold=self.iou_threshold
        )

        score = -1 * (
            (w * torch.log((1 - detections.conf) + 1e-9)).sum()
            # / len(detections)
        )

        return score


class TileWeightedFPCLoss(ObjectDetectionBaseLoss):

    def __call__(self, detections: DetectionsBase, ground_truthes: DetectionsBase):
        w = calc_det_weight(
            detections, ground_truthes, iou_threshold=self.iou_threshold
        )

        score = -1 * (
            ((1 - w) * torch.log(detections.conf + 1e-9)).sum()
            # / len(detections)
        )

        return score


def calc_det_weight(
    detections: DetectionsBase, ground_truthes: DetectionsBase, iou_threshold=0.5
):
    # iou_threshold = config.iou_threshold
    iou = list_iou(detections.xyxy, ground_truthes.xyxy)

    # 検出毎に正しい顔領域とのIOUをとり、閾値以上かそうでないかで二値化する
    z = torch.where((iou >= iou_threshold), 1, 0)

    # 検出毎に計算したIOUが閾値以上である正しい顔領域の数/正しい顔領域の合計数
    w = z.sum(dim=1) / len(ground_truthes)
    return w
