import torch


def image_crop_by_box(image, box):
    return image[:, :, int(box[1].item()):int(box[3].item())+1, int(box[0].item()):int(box[2].item())+1]


def box2mask(image, boxes):
    mask = torch.zeros(image.shape, device=image.device)
    for y in len(image.shape[2]):
        for x in len(image.shape[3]):
            if torch.loginal_and(torch.logical_and(boxes[:, 0] <= x, x <= boxes[:, 2]), torch.logical_and(boxes[:, 1] <= y, y <= boxes[:, 3])).any():
                mask[:, :, y, x] = 1
    return mask
