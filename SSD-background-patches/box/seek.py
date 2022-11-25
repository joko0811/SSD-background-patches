import torch


def smallest_box_containing(boxes):
    # box = xyxy
    min_x1y1, _ = boxes[..., :2].min(dim=0)
    max_x2y2, _ = boxes[..., 2:4].max(dim=0)

    return torch.cat((min_x1y1, max_x2y2))


def get_max_edge(boxes):
    # box = xywh
    return boxes[..., 2:4].max()


def find_nearest_box(box_listA, box_listB):
    """box_listAの各要素に対して最も近いbox_listBのインデックスを返す

    box=[xywh]
    # boxlist=[[x,y,w,h],[x,y,w,h],...]

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
