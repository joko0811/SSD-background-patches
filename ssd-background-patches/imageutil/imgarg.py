import torch


def augmentation_from_short_form_xywh(image, xyxy):
    """増強される画像は短形座標で示される画像の領域を移動させたもののリストを生成する
    Args:
        image: 増強元の画像
        xywh: 短形の座標
    """
    window_size = 25
    y_point = torch.round(
        torch.arange(0, image.shape[1], window_size) - xyxy[1] * image.shape[1]
    )
    x_point = torch.round(
        torch.arange(0, image.shape[2], window_size) - xyxy[0] * image.shape[2]
    )
    aug_shift_list = torch.cartesian_prod(y_point, x_point).to(
        device=image.device, dtype=torch.int
    )

    """
    # FIXME: out of memory
    augmented_image_list = torch.cat(
        [
            torch.roll(image, dims=(1, 2), shifts=aug_shift.tolist())
            for aug_shift in aug_shift_list
        ]
    )
    return augmented_image_list
    """
    return aug_shift_list
