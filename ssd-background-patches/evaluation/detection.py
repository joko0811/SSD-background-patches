import torch
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score


def ap(gt_box_list, det_box_list, det_conf, iou_threshold=0.5):
    """
    Args:
        gt_box_list: List of M_g*4 2D tensor. The number of elements in the list is equal to the number of images N. M is the number of ground truth detections per image.
        det_box_list: List of M_d*4 2D tensor. The number of elements in the list is equal to the number of images N. M is the number of detections per image.
        det_conf: N*M_d 2D tensor.
        iou_thresh: Threshold for determining TP.
    """
    # box=xywh

    tp_binary_list = list()
    for i, (gt_boxes, det_boxes) in enumerate(zip(gt_box_list, det_box_list)):
        tp_binary = (list_iou(det_boxes, gt_boxes) >= iou_threshold).any(
            dim=1).long()
        tp_binary_list.append(tp_binary.detach(
        ).cpu().resolve_conj().resolve_neg().tolist())

    ap_score = label_ranking_average_precision_score(tp_binary_list, det_conf)

    return ap_score


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
    """calculate list iou
    Args:
        box_listA: N*4 2D tensor
        box_listB: M*4 2D tensor
    Return:
        N*M iou list
    """
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


def data_utility_quority(ground_truth_det_num, tp, fp):
    return tp-fp/ground_truth_det_num


def calc_class_TP(det_boxes, gt_boxes, iou_threshold=0.5):

    tp_binary = (list_iou(det_boxes, gt_boxes) >=
                 iou_threshold).any(dim=1).long()

    tp_sum = tp_binary.sum().detach(
    ).cpu().resolve_conj().resolve_neg().tolist()

    return tp_sum


def calc_class_FP(det_boxes, gt_boxes, iou_threshold=0.5):
    """ground truthesと同じラベルを持ち、かつiouが閾値未満のdetectionの数をクラス別に集計する
    """
    fp_binary = (list_iou(det_boxes, gt_boxes) <
                 iou_threshold).any(dim=1).long()

    fp_sum = fp_binary.sum().detach(
    ).cpu().resolve_conj().resolve_neg().tolist()

    return fp_sum
