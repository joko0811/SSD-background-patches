import numpy as np

import torch
from detection.detection_base import DetectionsBase
from evaluation.detection import list_iou


class TpFpManager:
    def __init__(self, device):
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
        self.device = device

    def add_detection(
        self,
        detection: DetectionsBase,
        ground_truth: DetectionsBase,
        iou_thresh=0.5,
    ):
        """評価指標算出のための途中経過を記録する関数
        提案される顔領域を追加する
        同時にTP、FPなどの加算も行う
        Args:
            detection: 追加する検出
            ground_truth: 追加する検出に対応する真の顔領域。初期化段階で登録していない時に指定する
            iou_thresh: TP,FPの判定基準となるIoU閾値
        """
        gt_det, tp_det, fp_det, fn_det = self.judge_detection(
            detection, ground_truth, iou_thresh
        )
        det_conf_array, det_tp_binary_array = self.calc_sklearn_y_true_score(
            detection, ground_truth, iou_thresh
        )

        self.gt += len(gt_det)
        self.tp += len(tp_det)
        self.fp += len(fp_det)
        self.fn += len(fn_det)

        self.det_conf_array = np.append(self.det_conf_array, det_conf_array)
        self.det_tp_binary_array = np.append(
            self.det_tp_binary_array, det_tp_binary_array
        )

    def judge_detection(
        self,
        detection: DetectionsBase,
        ground_truth: DetectionsBase,
        iou_thresh=0.5,
    ):
        gt_det = torch.tensor([]).to(device=self.device)
        tp_det = torch.tensor([]).to(device=self.device)
        fp_det = torch.tensor([]).to(device=self.device)
        fn_det = torch.tensor([]).to(device=self.device)

        if (ground_truth is None) and (detection is not None):
            fp_det = torch.cat((detection.conf.unsqueeze(1), detection.xyxy), dim=1)

        elif (ground_truth is not None) and (detection is None):
            gt_det = torch.cat(
                (ground_truth.conf.unsqueeze(1), ground_truth.xyxy), dim=1
            )
            fn_det = torch.cat(
                (ground_truth.conf.unsqueeze(1), ground_truth.xyxy), dim=1
            )

        elif (ground_truth is not None) and (detection is not None):
            gt_det = torch.cat(
                (ground_truth.conf.unsqueeze(1), ground_truth.xyxy), dim=1
            )

            dt_gt_iou = list_iou(detection.xyxy, ground_truth.xyxy)

            det_tp_binary = (dt_gt_iou >= iou_thresh).any(dim=1)
            det_fp_binary = (dt_gt_iou < iou_thresh).all(dim=1)
            det_fn_binary = (dt_gt_iou < iou_thresh).all(dim=0)

            # 重なる検出が一つもない真の領域数を集計
            # all(dim=0)で真の領域毎の重なる領域が一つもない場合Trueを取るboolに変換
            # detach以降はtensor→intの変換処理
            tp_det = torch.cat(
                (
                    detection.conf[det_tp_binary].unsqueeze(1),
                    detection.xyxy[det_tp_binary],
                ),
                dim=1,
            )
            fp_det = torch.cat(
                (
                    detection.conf[det_fp_binary].unsqueeze(1),
                    detection.xyxy[det_fp_binary],
                ),
                dim=1,
            )
            fn_det = torch.cat(
                (
                    ground_truth.conf[det_fn_binary].unsqueeze(1),
                    ground_truth.xyxy[det_fn_binary],
                ),
                dim=1,
            )

        return gt_det, tp_det, fp_det, fn_det

    def calc_sklearn_y_true_score(self, detection, ground_truth, iou_thresh=0.5):
        det_conf = np.array([])
        det_tp_binary = np.array([])

        if (ground_truth is not None) and (detection is not None):
            dt_gt_iou = list_iou(detection.xyxy, ground_truth.xyxy)

            det_tp_binary = (
                (dt_gt_iou >= iou_thresh)
                .any(dim=1)
                .long()
                .detach()
                .cpu()
                .resolve_conj()
                .resolve_neg()
                .numpy()
            )

            det_conf = (
                detection.conf.detach().cpu().resolve_conj().resolve_neg().numpy()
            )

        elif detection is not None:
            det_tp_binary = np.zeros(len(detection))

            det_conf = (
                detection.conf.detach().cpu().resolve_conj().resolve_neg().numpy()
            )

        return det_conf, det_tp_binary

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
