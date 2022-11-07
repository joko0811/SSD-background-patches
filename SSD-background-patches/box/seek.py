import torch


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
