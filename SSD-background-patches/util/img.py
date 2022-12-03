import cv2
import numpy as np

from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def save_tensor_image(image, image_path):
    pil_img = transforms.functional.to_pil_image(image[0])
    pil_img.save(image_path)


def pil2cv(pil_image):
    new_image = np.array(pil_image, dtype=np.uint8)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def cv2pil(image):
    new_image = image.copy()
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def tensor2annotation_image(image, detections, class_names):
    pil_img = transforms.functional.to_pil_image(image[0])

    ann_img = draw_annotations(pil_img, detections.xyxy,
                               detections.class_labels, detections.confidences, class_names)
    return ann_img

# https://pystyle.info/pillow-draw-object-detection-results-on-an-image/


def draw_annotations(img, boxes, class_labeles, confidences, class_names):
    """画像に対してアノテーションを追加する

    Args:
        img:
            PIL type image
        boxes:
            xyxy

    Returns:
        pil type image:
            アノテーションを追加した画像
    """

    draw = ImageDraw.Draw(img, mode="RGBA")

    # 色の一覧を作成する。
    cmap = plt.cm.get_cmap("hsv", len(class_names) + 1)

    # フォントを作成する。
    fontsize = max(15, int(0.03 * min(img.size)))
    fontname = "DejaVuSerif-Bold"
    font = ImageFont.truetype(fontname, size=fontsize)

    for i in range(len(boxes)):
        # 色を取得する。
        color = cmap(int(class_labeles[i].item()), bytes=True)

        # ラベル
        caption = class_names[class_labeles[i].int()]
        caption += f" {confidences[i]:.0%}"

        # 矩形を描画する。
        draw.rectangle(
            boxes[i].tolist(), outline=color, width=3
        )

        # ラベルを描画する。
        text_w, text_h = draw.textsize(caption, font=font)
        text_x2 = boxes[i, 0] + text_w - 1
        text_y2 = boxes[i, 1] + text_h - 1

        draw.rectangle(
            (boxes[i, 0].item(), boxes[i, 1].item(), text_x2.item(), text_y2.item()), fill=color)
        draw.text((boxes[i, 0].item(), boxes[i, 1].item()),
                  caption, fill="black", font=font)

    return img


def tensor2box_annotation_image(image, boxes):
    pil_img = transforms.functional.to_pil_image(image[0])
    box_img = draw_boxes(pil_img, boxes)
    return box_img


def draw_boxes(img, boxes):

    draw = ImageDraw.Draw(img, mode="RGBA")

    for i in range(len(boxes)):

        # 矩形を描画する。
        draw.rectangle(
            boxes[i].tolist(), width=3
        )

    return img
