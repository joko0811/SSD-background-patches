import rembg  # tool to remove images background

import torch

from imageutil import imgconv


def gen_mask_image(image: torch.Tensor):
    """Returns the mask image of image
    Args:
        image: 4D tensor N*H*W*C or 3D tensor H*W*C.
    Return: tensor of the same shape as the input
    """
    device = image.device

    if image.dim() == 3:
        # This conversion is done because rembg input is a pil image
        pil_image = imgconv.tensor2pil(image)
        return imgconv.pil2tensor(rembg.remove(pil_image, only_mask=True), device)

    mask_image_list = torch.zeros(
        image.shape, device=device, dtype=torch.float)
    for image_iter, img in enumerate(image):
        pil_image = imgconv.tensor2pil(img)
        mask_image = rembg.remove(pil_image, only_mask=True)
        mask_image_list[image_iter] = imgconv.pil2tensor(
            mask_image, device)

    return mask_image_list


def composite_image(foreground_image: torch.Tensor, background_image: torch.Tensor, mask_image: torch.Tensor):
    """Compose an image based on a mask image

    Args:
        foreground_image: 4D tensor N*H*W*C or 3D tensor H*W*C. Same shape as mask_image
        background_image: 4D tensor N*H*W*C or 3D tensor H*W*C
        mask_image: 4D tensor N*H*W*C or 3D tensor H*W*C. Same shape as foreground_image. Background is (0,0,0)
    Return: 4D tensor N*H*W*C or 3D tensor H*W*C
    """

    # Match the shape of background_image
    if background_image.dim() == 3:
        composite_background_image = background_image.tile(
            (foreground_image.shape[0], 1, 1, 1))
    else:
        composite_background_image = background_image.clone()

    composite_image = torch.where(
        mask_image > 0, foreground_image, composite_background_image)
    return composite_image
