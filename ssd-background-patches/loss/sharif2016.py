import torch


def tv_loss(image_list, gradation_strength=0.1):
    """
    Args:
        image_list: N*C*H*W
        gradation_strength: The closer the value is to 1, the more monochromatic the patch, and the closer it is to 0, the more colorful the patch.
    """

    roll_h_image = torch.roll(input=image_list, shifts=(0, 0, 1, 0), dims=(0, 1, 2, 3))
    roll_h_image[:, :, 0, :] = 0

    roll_w_image = torch.roll(input=image_list, shifts=(0, 0, 0, 1), dims=(0, 1, 2, 3))
    roll_w_image[:, :, :, 0] = 0

    image_h_diff = torch.abs(image_list - roll_h_image)
    image_w_diff = torch.abs(image_list - roll_w_image)
    tv_score = (
        torch.sum(torch.sqrt(torch.pow(image_h_diff, 2) + torch.pow(image_w_diff, 2)))
        / gradation_strength
        * (image_list.shape[2] * image_list.shape[3])
    )

    return tv_score


def nps_loss(image_list):
    return
