from detection.detection_base import DetectionsBase


class ObjectDetectionBaseLoss:
    def __init__(
        self,
        iou_threshold=0.5,
        contain_nms=False,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.contain_nms = contain_nms

    def __call__(self, detections: DetectionsBase, ground_truthes: DetectionsBase):
        # NOTE: Post-processing of output (NMS, etc.) could be done here in the future.
        pass
