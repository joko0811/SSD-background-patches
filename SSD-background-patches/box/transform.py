import torch


def image_crop_by_box(image, box):
    # box=xyxy
    return image[:, :, int(box[1].item()):int(box[3].item())+1, int(box[0].item()):int(box[2].item())+1]


def box2mask(image, boxes):
    mask = torch.zeros(image.shape, device=image.device)
    for y in range(image.shape[2]):
        for x in range(image.shape[3]):
            if torch.logical_and(torch.logical_and(boxes[:, 0] <= x, x <= boxes[:, 2]), torch.logical_and(boxes[:, 1] <= y, y <= boxes[:, 3])).any():
                mask[:, :, y, x] = 1
    return mask


def box_expand(boxes, offset):
    # box=xyxy
    x1y1 = boxes[..., :2]-offset
    x2y2 = boxes[..., 2:4]+offset

    dim = 1
    if boxes.dim() == 1:
        dim = 0
    return torch.cat((x1y1, x2y2), dim=dim)


def box_clamp(boxes, w, h):
    x1 = torch.clamp(boxes[..., 0], 0, w)
    y1 = torch.clamp(boxes[..., 1], 0, h)
    x2 = torch.clamp(boxes[..., 2], 0, w)
    y2 = torch.clamp(boxes[..., 3], 0, h)

    if boxes.dim() == 1:
        return torch.tensor([x1, y1, x2, y2], device=x1.device)
    else:
        return torch.cat((x1, y1, x2, y2), dim=1)
