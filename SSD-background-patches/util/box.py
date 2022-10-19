

def xywh2x1y1(box):
    """xywh to lrtb(x1y1x2y2)
    """
    center_x, center_y, width, height = box

    left = center_x-(width/2)
    right = center_x+(width/2)
    top = center_y-(height/2)
    bottom = center_y+(height/2)
    return [left, right, top, bottom]


def lrtb2xywh(box):
    """lrtb(x1y1x2y2) to xywh
    """

    left, right, top, bottom = box

    x = (left+right)/2
    y = (top+bottom)/2
    w = abs(right-left)
    h = abs(bottom-top)
    return [x, y, w, h]


def is_overlap(boxA, boxB) -> bool:
    # box = lrtb(x1y1x2y2)
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
