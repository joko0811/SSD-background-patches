import torch


def image_crop_by_box(image, box):
    return image[:, :, int(box[1].item()):int(box[3].item())+1, int(box[0].item()):int(box[2].item())+1]
