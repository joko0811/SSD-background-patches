import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import ImageDraw, ImageFont


def pil2cv(pil_image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(pil_image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


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


def draw_boxes(img, boxes):

    draw = ImageDraw.Draw(img, mode="RGBA")

    for i in range(len(boxes)):

        # 矩形を描画する。
        draw.rectangle(
            boxes[i].tolist(), width=3
        )

    return img
