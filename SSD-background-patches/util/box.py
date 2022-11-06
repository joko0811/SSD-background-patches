import torch


def xywh2xyxy(boxes):
    """xywh to lrtb(x1y1x2y2)
    """
    x = boxes[..., 0]
    y = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]

    x1 = x-(w/2)
    y1 = y-(h/2)
    x2 = x+(w/2)
    y2 = y+(h/2)

    return torch.cat([x1, y1, x2, y2], dim=1)


def xyxy2xywh(boxes):
    """lrtb(x1y1x2y2) to xywh
    """

    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    x2 = boxes[..., 2]
    y2 = boxes[..., 3]

    x = (x1+x2)/2
    y = (y1+y2)/2
    w = abs(x2-x1)
    h = abs(y2-y1)
    return torch.cat([x, y, w, h], dim=1)


def is_overlap(boxA, boxB) -> bool:
    """return True if
    箱どうしが重なっているか判定する
    """
    # box = lrtb (x1y1x2y2)
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return (max(ax1, bx1) <= min(ax2, bx2)) and (max(ay1, by1) <= min(ay2, by2))


def get_max_edge(box):
    # box = lrtb (x1y1x2y2)
    return max([abs(box[0]-box[2]), abs(box[1]-box[3])])


def find_nearest_box(box_listA, box_listB):
    """box_listAの各要素に対して最も近いbox_listBのインデックスを返す

    box=[xywh]

    Returns:
        nearest_idx:
            box_listAの要素数に等しい
    """

    nearest_idx = torch.zeros((box_listA.shape[0]), device=box_listA.device)
    for i, boxA in enumerate(box_listA):
        norm = torch.linalg.norm(box_listB[:, :2]-boxA[:2], dim=1)
        min_idx = torch.argmin(norm)
        nearest_idx[i] = min_idx

    return nearest_idx.to(torch.int64)


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
