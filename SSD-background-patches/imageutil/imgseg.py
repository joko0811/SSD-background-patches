import rembg  # tool to remove images background
from torchvision import transforms


def gen_mask_image(image):
    return rembg.remove(image, only_mask=True)


def composite_image(foreground_image, background_image, mask_image, pos):
    """PIL画像の合成
    """
    im = background_image.copy()
    im.paste(foreground_image, pos, mask_image)
    return im


def wrap_composite_image(foreground_image, background_image, mask_image):
    """合成位置中心固定
    Args:
        foreground_image: tensor image
        background_image: tensor image
        mask_image: pil image
    Return:
        im: tensor image
    """
    fg = transforms.functional.to_pil_image(foreground_image)
    bg = transforms.functional.to_pil_image(background_image)

    x = abs(int(background_image.shape[1]/2 - foreground_image.shape[1]/2))
    y = abs(int(background_image.shape[0]/2 - foreground_image.shape[0]/2))
    pos = [x, y]

    im = composite_image(fg, bg, mask_image, pos)
    return transforms.functional.to_tensor(im)
