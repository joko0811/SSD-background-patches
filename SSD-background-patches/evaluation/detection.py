import torch


def iou(boxA, boxB):
    # box = lrtb (x1y1x2y2)
    ax1 = boxA[:, 0]
    ay1 = boxA[:, 1]
    ax2 = boxA[:, 2]
    ay2 = boxA[:, 3]
    bx1 = boxB[:, 0]
    by1 = boxB[:, 1]
    bx2 = boxB[:, 2]
    by2 = boxB[:, 3]

    intersect = (torch.min(ax2, bx2)-torch.max(ax1, bx1)) * \
        (torch.min(ay2, by2)-torch.max(ay1, by1))
    a_area = (ax2-ax1)*(ay2-ay1)
    b_area = (bx2-bx1)*(by2-by1)

    iou = intersect/(a_area+b_area-intersect)
    return iou


def list_iou(box_listA, box_listB):
    # box = xyxy
    compare_A = box_listA.unsqueeze(1)
    compare_B = box_listB.unsqueeze(0)

    ax1 = compare_A[..., 0]
    ay1 = compare_A[..., 1]
    ax2 = compare_A[..., 2]
    ay2 = compare_A[..., 3]
    bx1 = compare_B[..., 0]
    by1 = compare_B[..., 1]
    bx2 = compare_B[..., 2]
    by2 = compare_B[..., 3]

    intersect = (torch.min(ax2, bx2)-torch.max(ax1, bx1)) * \
        (torch.min(ay2, by2)-torch.max(ay1, by1))
    a_area = (ax2-ax1)*(ay2-ay1)
    b_area = (bx2-bx1)*(by2-by1)

    iou = intersect/(a_area+b_area-intersect)
    return iou


def accuracy(tp, tn, fp, fn):
    return (tp+tn)/(tp+tn+fp+fn)


def precision(tp, fp):
    return tp/(tp+fp)


def recall(tp, fn):
    return tp/(tp+fn)


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
