import torch


def mean_average_precision():
    return


def calc_class_TP(detections, ground_truthes, total_class, iou_scores, iou_threshold=0.5):
    """ground truthesと同じラベルを持ち、かつiouが閾値以上のdetectionの数をクラス別に集計する
    """
    det_labels = detections.class_labels.unsqueeze(0)
    gt_labels = ground_truthes.class_labels.unsqueeze(1)
    label_flag = (det_labels == gt_labels)

    iou_flag = (iou_scores >= iou_threshold)

    # 検出毎のTP条件を満たすフラグ
    tp_flag = torch.logical_and(label_flag, iou_flag).T.any(dim=1)

    # TP条件を満たす検出数をクラスラベル毎に集計
    aggregation_TP_by_class = torch.zeros(
        total_class, device=tp_flag.device, dtype=torch.int64).scatter_add(0, detections.class_labels, tp_flag.to(torch.int64))
    return aggregation_TP_by_class


def calc_class_FP(detections, ground_truthes, total_class, iou_scores, iou_threshold=0.5):
    """ground truthesと同じラベルを持ち、かつiouが閾値未満のdetectionの数をクラス別に集計する
    """

    det_labels = detections.class_labels.unsqueeze(0)
    gt_labels = ground_truthes.class_labels.unsqueeze(1)
    label_flag = (det_labels == gt_labels)

    iou_flag = (iou_scores < iou_threshold)

    # 検出毎のFP条件を満たすフラグ
    fp_flag = torch.logical_and(label_flag, iou_flag).T.any(dim=1)

    # FP条件を満たす検出数をクラスラベル毎に集計
    aggregation_FP_by_class = torch.zeros(
        total_class, device=fp_flag.device, dtype=torch.int64).scatter_add(0, detections.class_labels, fp_flag.to(torch.int64))
    return aggregation_FP_by_class


def calc_class_FN(ground_truthes, total_class, iou_scores,  iou_threshold=0.5):
    """どのdetectionsともiouが閾値以下のground truthの数をクラス別に集計する
    """
    fn_flag = (iou_scores < iou_threshold).any(dim=1)

    # FP条件を満たす検出数をクラスラベル毎に集計
    aggregation_FN_by_class = torch.zeros(
        total_class, device=fn_flag.device, dtype=torch.int64).scatter_add(0, ground_truthes.class_labels, fn_flag.to(torch.int64))
    return aggregation_FN_by_class
