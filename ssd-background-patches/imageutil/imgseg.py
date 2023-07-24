import rembg  # tool to remove images background
from PIL import Image

import torch


def generate_mask_image(image: Image.Image):
    mask_image = rembg.remove(image, only_mask=True)
    return mask_image


def composite_image(
    foreground_image: torch.Tensor,
    background_image: torch.Tensor,
    mask_image: torch.Tensor,
):
    """Compose an image based on a mask image

    Args:
        foreground_image: 4D tensor N*H*W*C or 3D tensor H*W*C. Same shape as mask_image
        background_image: 4D tensor N*H*W*C or 3D tensor H*W*C
        mask_image: 4D tensor N*H*W*C or 3D tensor H*W*C. Same shape as foreground_image. Background is (0,0,0)
    Return: 4D tensor N*H*W*C or 3D tensor H*W*C
    """

    # Match the shape of background_image
    if background_image.dim() == 3:
        composite_background_image = background_image.repeat(
            (foreground_image.shape[0], 1, 1, 1)
        )
    else:
        composite_background_image = background_image.clone()

    if mask_image.shape[1] != 3:
        composite_mask_image = mask_image.repeat(1, 3, 1, 1)
    else:
        composite_mask_image = mask_image.clone()

    composite_image = torch.where(
        composite_mask_image, foreground_image, composite_background_image
    )
    return composite_image


def composite_image_with_3_layer(
    fg_bg_image, fg_bg_mask_image, mg_image, mg_mask_image
):
    """二つの画像を前景、中景、背景の三つのレイヤーで合成する
    Args:
        fg_bg_image: 前景と背景を含む画像
        bg_mask_image: 前傾が1、背景が0
        mg_image: 中景を含む画像
        mg_mask_image: 中景が1、破棄する領域が0
    """

    if mg_image.dim() == 3:
        composite_mg_image = mg_image.repeat((fg_bg_image.shape[0], 1, 1, 1))
    else:
        composite_mg_image = mg_image.clone()

    if mg_mask_image.dim() == 3:
        composite_mg_mask_image = mg_mask_image.repeat((fg_bg_image.shape[0], 1, 1, 1))
    else:
        composite_mg_mask_image = mg_mask_image.clone()

    if fg_bg_mask_image.shape[1] != 3:
        composite_mg_mask_image = composite_mg_mask_image.repeat(1, 3, 1, 1)
    else:
        composite_mg_mask_image = composite_mg_mask_image.clone()

    if fg_bg_mask_image.shape[1] != 3:
        composite_fg_bg_mask_image = fg_bg_mask_image.repeat(1, 3, 1, 1)
    else:
        composite_fg_bg_mask_image = fg_bg_mask_image.clone()

    composite_image = torch.where(
        (
            torch.logical_and(
                torch.logical_not(composite_fg_bg_mask_image), composite_mg_mask_image
            )
        ),
        composite_mg_image,
        fg_bg_image,
    )
    return composite_image
