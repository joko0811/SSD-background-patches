import numpy as np

from detection.detection_base import DetectionsBase
from evaluation.detection import list_iou


class TpFpManager:
    def __init__(self, ground_truth=None):
        self.ground_truth = ground_truth
        self.tp = 0
        self.fp = 0
        self.gt = 0 if self.ground_truth is None else len(self.ground_truth)
        self.det_tp_binary_array = np.array([])
        self.det_conf_array = np.array([])

    def add_detection(
        self,
        detection: DetectionsBase,
        ground_truth: DetectionsBase = None,
        iou_thresh=0.5,
    ):
        if self.ground_truth is None:
            if (ground_truth is None) and (detection is None):
                return

            elif (ground_truth is None) and (detection is not None):
                self.det_tp_binary_array = np.append(
                    self.det_tp_binary_array, np.zeros(len(detection))
                )
                self.det_conf_array = np.append(
                    self.det_conf_array,
                    detection.conf.detach().cpu().resolve_conj().resolve_neg().numpy(),
                )
                self.fp += len(detection)
                return

            elif (ground_truth is not None) and (detection is None):
                self.gt += len(ground_truth)
                return

            elif (ground_truth is not None) and (detection is not None):
                self.gt += len(ground_truth)

                det_tp_binary = (
                    (list_iou(detection.xyxy, ground_truth.xyxy) >= iou_thresh)
                    .any(dim=1)
                    .long()
                )

                self.tp += det_tp_binary.nonzero().shape[0]
                self.fp += det_tp_binary.shape[0] - det_tp_binary.nonzero().shape[0]

                self.det_tp_binary_array = np.append(
                    self.det_tp_binary_array,
                    det_tp_binary.detach().cpu().resolve_conj().resolve_neg().numpy(),
                )
                self.det_conf_array = np.append(
                    self.det_conf_array,
                    detection.conf.detach().cpu().resolve_conj().resolve_neg().numpy(),
                )
                return
        else:
            if detection is None:
                return

            det_tp_binary = (
                (list_iou(detection.xyxy, self.ground_truth.xyxy) >= iou_thresh)
                .any(dim=1)
                .long()
            )

            self.tp += det_tp_binary.nonzero().shape[0]
            self.fp += det_tp_binary.shape[0] - det_tp_binary.nonzero().shape[0]

            self.det_tp_binary_array = np.append(
                self.det_tp_binary_array,
                det_tp_binary.detach().cpu().resolve_conj().resolve_neg().numpy(),
            )
            self.det_conf_array = np.append(
                self.det_conf_array,
                detection.conf.detach().cpu().resolve_conj().resolve_neg().numpy(),
            )
        return

    def reset(self):
        self.reset_det()
        self.reset_sklean_y_true_score()

    def reset_det(self):
        self.tp = 0
        self.fp = 0
        self.gt = 0 if self.ground_truth is None else len(self.ground_truth)

    def reset_sklean_y_true_score(self):
        self.det_tp_binary_array = np.array([])
        self.det_conf_array = np.array([])

    def get_det(self):
        return (self.tp, self.fp, self.gt)

    def get_sklearn_y_true_score(self):
        return (self.det_tp_binary_array, self.det_conf_array)
