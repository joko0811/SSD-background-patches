import numpy as np

import torch
from torchvision import transforms

import cv2
from PIL import Image


def image_clamp(image, min, max):
    """
    Args:
        image: CHW 3D tensor(RGB)
        min: (RGB) min
        max: (RGB) max
    Return:
        clamped CHW 3D tensor(RGB)
    """
    imageR = image[0, :, :].detach().clone()
    clamped_imageR = torch.clamp(imageR, min=min[0], max=max[0]).unsqueeze(0)
    imageG = image[1, :, :].detach().clone()
    clamped_imageG = torch.clamp(imageG, min=min[1], max=max[1]).unsqueeze(0)
    imageB = image[2, :, :].detach().clone()
    clamped_imageB = torch.clamp(imageB, min=min[2], max=max[2]).unsqueeze(0)

    clamped_image = torch.cat(
        (clamped_imageR, clamped_imageG, clamped_imageB), dim=0
    ).contiguous()

    return clamped_image


def video2image(video_path, sample_interbal=1):
    # 動作未確認
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # error
        return

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

    # 画像枚数分のリスト
    image_list = np.zeros(
        (int(frame_count / sample_interbal), frame_height, frame_width, 3)
    )

    for i in range(frame_count):
        if i % sample_interbal == 0:
            image_list[i] = cap.read()

    cap.release()
    return image_list


def cv2pil(image):
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def pil2tensor(image, device):
    return transforms.functional.to_tensor(image).to(device=device, dtype=torch.float)


def tensor2pil(image):
    if image.dim == 3:
        return transforms.functional.to_pil_image(image)
    else:
        image_list = list()
        for img in image:
            image_list.append(transforms.functional.to_pil_image(img))
        return image_list
