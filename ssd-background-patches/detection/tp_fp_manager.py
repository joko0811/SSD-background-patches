import numpy as np

from detection.detection_base import DetectionsBase
from evaluation.detection import list_iou


class TpFpManager:
    def __init__(self):
        """
        Args:
            ground_truth: 真の顔領域を示す検出。初期化段階で全ての真の顔領域が既知である時に指定する
        """
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.gt = 0
        self.det_tp_binary_array = np.array([])
        self.det_conf_array = np.array([])

    def add_detection(
        self,
        detection: DetectionsBase,
        ground_truth: DetectionsBase,
        iou_thresh=0.5,
    ):
        """
        提案される顔領域を追加する
        同時にTP、FPなどの加算も行う
        Args:
            detection: 追加する検出
            ground_truth: 追加する検出に対応する真の顔領域。初期化段階で登録していない時に指定する
            iou_thresh: TP,FPの判定基準となるIoU閾値
        """
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
            self.fn += len(ground_truth)
            return

        elif (ground_truth is not None) and (detection is not None):
            self.gt += len(ground_truth)

            dt_gt_iou = list_iou(detection.xyxy, ground_truth.xyxy)

            det_tp_binary = (
                (dt_gt_iou >= iou_thresh)
                .any(dim=1)
                .long()
            )

            # 重なる検出が一つもない真の領域数を集計
            # any(dim=0)で真の領域毎の重なる領域が存在するかのboolに変換
            # detach以降はtensor→intの変換処理
            self.fn += ((dt_gt_iou < iou_thresh).any(dim=0).long().sum()
                        ).detach().cpu().resolve_conj().resolve_neg().numpy().item()

            self.tp += det_tp_binary.nonzero().shape[0]
            self.fp += det_tp_binary.shape[0] - \
                det_tp_binary.nonzero().shape[0]

            self.det_tp_binary_array = np.append(
                self.det_tp_binary_array,
                det_tp_binary.detach().cpu().resolve_conj().resolve_neg().numpy(),
            )
            self.det_conf_array = np.append(
                self.det_conf_array,
                detection.conf.detach().cpu().resolve_conj().resolve_neg().numpy(),
            )
            return

    def get_value(self):
        return (self.tp, self.fp, self.fn, self.gt)

    def get_sklearn_y_true_score(self):
        """
        sklearn.metrics.average_precision_scoreなどで引数として使用される値を返す
        Return:
            det_tp_binary_array: 検出毎のTP判定の二値変数（0or1）
            det_conf_array: 検出毎の信頼値
        """
        return (self.det_tp_binary_array, self.det_conf_array)
