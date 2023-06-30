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

    intersect = (torch.min(ax2, bx2) - torch.max(ax1, bx1)).clamp(0) * (
        torch.min(ay2, by2) - torch.max(ay1, by1)
    ).clamp(0)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)

    iou = intersect / (a_area + b_area - intersect + 1e-9)
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

    intersect = (torch.min(ax2, bx2) - torch.max(ax1, bx1)).clamp(0) * (
        torch.min(ay2, by2) - torch.max(ay1, by1)
    ).clamp(0)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)

    iou = intersect / (a_area + b_area - intersect + 1e-9)
    return iou


def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def precision(tp, fp):
    return tp / (tp + fp + 1e-9)


def recall(tp, fn):
    return tp / (tp + fn + 1e-9)


def data_utility_quority(ground_truth_det_num, tp, fp):
    return (tp - fp) / (ground_truth_det_num + 1e-9)


def f1(precision, recall, beta=1):
    return (1 + beta) * precision * recall / (beta * precision + recall + 1e-9)
