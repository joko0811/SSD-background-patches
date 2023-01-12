from PIL import ImageDraw, ImageFont

from matplotlib import pyplot as plt
from torchvision import transforms

from box.boxio import detections_base


# https://pystyle.info/pillow-draw-object-detection-results-on-an-image/
def draw_annotations(image, detections: detections_base, class_names, in_confidences=True):
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
    pil_image = transforms.functional.to_pil_image(image.clone().detach())

    draw = ImageDraw.Draw(pil_image, mode="RGBA")

    # 色の一覧を作成する。
    cmap = plt.cm.get_cmap("hsv", len(class_names) + 1)

    # フォントを作成する。
    fontsize = max(15, int(0.03 * min(pil_image.size)))
    fontname = "DejaVuSerif-Bold"
    font = ImageFont.truetype(fontname, size=fontsize)

    for i in range(len(detections.xyxy)):
        # 色を取得する
        color = cmap(int(detections.class_labels[i].item()), bytes=True)

        # ラベル
        caption = class_names[detections.class_labels[i].int()]
        if in_confidences:
            caption += f" {detections.confidences[i]:.0%}"

        # 矩形を描画する
        draw.rectangle(
            detections.xyxy[i].tolist(), outline=color, width=3
        )

        # ラベルを描画する
        text_w, text_h = draw.textsize(caption, font=font)
        text_x2 = detections.xyxy[i, 0] + text_w - 1
        text_y2 = detections.xyxy[i, 1] + text_h - 1

        draw.rectangle(
            (detections.xyxy[i, 0].item(), detections.xyxy[i, 1].item(), text_x2.item(), text_y2.item()), fill=color)
        draw.text((detections.xyxy[i, 0].item(), detections.xyxy[i, 1].item()),
                  caption, fill="black", font=font)

    return transforms.functional.to_tensor(pil_image)


def draw_boxes(image, boxes):
    pil_image = transforms.functional.to_pil_image(image.clone().detach())

    draw = ImageDraw.Draw(pil_image, mode="RGBA")

    for i in range(len(boxes)):

        # 矩形を描画する
        draw.rectangle(
            boxes[i].tolist(), width=3
        )

    return transforms.functional.to_tensor(pil_image)
