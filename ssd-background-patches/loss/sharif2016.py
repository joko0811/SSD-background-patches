import torch


def tv_loss(image_list):
    """
    Args:
        image_list: N*C*H*W
    """

    roll_h_image = torch.roll(input=image_list, shifts=(
        0, 0, 1, 0), dims=(0, 1, 2, 3))
    roll_h_image[:, :, 0, :] = 0

    roll_w_image = torch.roll(input=image_list, shifts=(
        0, 0, 0, 1), dims=(0, 1, 2, 3))
    roll_w_image[:, :, :, 0] = 0

    tv_score = torch.sum(torch.sqrt(
        torch.pow(image_list-roll_h_image, 2)+torch.pow(image_list-roll_w_image, 2)))

    return tv_score


def nps_loss(image_list):
    return
