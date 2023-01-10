import torch


def image_crop_by_box(image, box):
    # box=xyxy
    return image[:, :, int(box[1].item()):int(box[3].item())+1, int(box[0].item()):int(box[2].item())+1]


def box2mask(image, boxes):

    image_h = image.shape[2]
    image_w = image.shape[3]
    mask_size = (image_h, image_w)

    # マスクしたい領域が0
    mask = torch.ones(mask_size, device=image.device)

    for box in boxes:
        x1, y1, x2, y2 = box

        sc_idx_y = (torch.arange(y2-y1, dtype=torch.int64, device=image.device) +
                    y1.to(torch.int64)).unsqueeze(1).tile(image_w)
        sc_src_y = torch.ones(sc_idx_y.shape, device=sc_idx_y.device)
        mask_y = torch.zeros(mask_size, device=image.device).scatter_(
            0, sc_idx_y, sc_src_y)

        sc_idx_x = (torch.arange(x2-x1, dtype=torch.int64, device=image.device) +
                    x1.to(torch.int64)).tile(image_h, 1)
        sc_src_x = torch.ones(sc_idx_x.shape, device=sc_idx_x.device)
        mask_x = torch.zeros(mask_size, device=image.device).scatter_(
            1, sc_idx_x, sc_src_x)

        # mask_y,mask_x共通して1である箇所は0、それ以外は1のbox_mask
        box_mask = torch.logical_not(mask_y*mask_x).int()

        mask *= box_mask

    # 0->1,1->0 マスクしたい領域が1
    return (mask == 0).int()


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
