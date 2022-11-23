import torch


def is_overlap(boxA, boxB) -> bool:
    """return True if
    箱どうしが重なっているか判定する
    """
    # box = lrtb (x1y1x2y2)
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return (max(ax1, bx1) <= min(ax2, bx2)) and (max(ay1, by1) <= min(ay2, by2))


def is_overlap_list(boxA, box_listB):
    """return True if
    箱どうしが重なっているか判定する
    """
    # box = lrtb (x1y1x2y2)
    ax1, ay1, ax2, ay2 = boxA
    bx1 = box_listB[..., 0]
    by1 = box_listB[..., 1]
    bx2 = box_listB[..., 2]
    by2 = box_listB[..., 3]

    return torch.logical_and((torch.max(ax1, bx1) <= torch.min(ax2, bx2)), (torch.max(ay1, by1) <= torch.min(ay2, by2))).all()


def xywh2xyxy(xywh):
    xyxy = xywh.new(xywh.shape)
    # x1=x-w/2
    xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2
    # y1=y-h/2
    xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2
    # x2=x+w/2
    xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2
    # y2=y+h/2
    xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2
    return xyxy


def xyxy2xywh(xyxy):
    xywh = xyxy.new(xyxy.shape)
    # x=(x1+x2)/2
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
    # y=(y1+y2)/2
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
    # w=|x2-x1|
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    # h=|y2-y1|
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return xywh


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
