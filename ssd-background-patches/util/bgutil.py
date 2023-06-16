# background util

from imageutil import imgseg


def background_applyer(foreground_image, background_image):
    mask_image = imgseg.gen_mask_image(foreground_image)
    applied_image = imgseg.composite_image(
        foreground_image, background_image, mask_image
    )
    return applied_image
