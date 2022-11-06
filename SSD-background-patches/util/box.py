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
    if boxA == boxB:
        return True
    elif (
        (ax1 <= bx1 and ax2 > bx1) or (ax1 >= bx1 and bx2 > ax1)
    ) and (
        (ay1 <= by1 and ay2 > by1) or (ay1 >= by1 and by2 > ay1)
    ):
        return True
    else:
        return False


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

    for boxA in box_listA:
        min_idx = torch.argmin([torch.linalg.norm(boxA[:2]-bb[:2])
                                for bb in box_listB])
        nearest_idx = torch.cat([nearest_idx, min_idx])

    return nearest_idx


def iou(boxA, boxB):
    # box = lrtb (x1y1x2y2)
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    intersect = (min(ax2, bx2)-max(ax1, bx1))*(min(ay2, by2)-max(ay1, by1))
    a_area = (ax2-ax1)*(ay2-ay1)
    b_area = (bx2-bx1)*(by2-by1)

    iou = intersect/(a_area+b_area-intersect)
    return iou
