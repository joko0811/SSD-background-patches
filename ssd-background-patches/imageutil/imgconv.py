import numpy as np

import torch
from torchvision import transforms

import cv2
from PIL import Image


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
    image_list = np.zeros((int(frame_count/sample_interbal),
                          frame_height, frame_width, 3))

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
    return transforms.functional.to_pil_image(image)
