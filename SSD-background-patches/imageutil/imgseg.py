import rembg  # tool to remove images background


def gen_mask_image(image):
    return rembg.remove(image, only_mask=True)


def composite_images(foreground_image, background_image, mask_image, pos):
    """PIL画像の合成
    """
    im = background_image.copy()
    im.paste(foreground_image, pos, mask_image)
    return im
