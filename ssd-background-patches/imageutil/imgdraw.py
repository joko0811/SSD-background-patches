from PIL import ImageDraw, ImageFont

from matplotlib import pyplot as plt


from detection.detection_base import ObjectDetectionBase

# https://pystyle.info/pillow-draw-object-detection-results-on-an-image/


def draw_annotations(
    image, detections: ObjectDetectionBase, class_names, in_confidences=True
):
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

    draw = ImageDraw.Draw(image, mode="RGBA")

    # 色の一覧を作成する。
    cmap = plt.cm.get_cmap("hsv", len(class_names) + 1)

    # フォントを作成する。
    fontsize = max(15, int(0.03 * min(image.size)))
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
        draw.rectangle(detections.xyxy[i].tolist(), outline=color, width=3)

        # ラベルを描画する
        text_w, text_h = draw.textsize(caption, font=font)
        text_x2 = detections.xyxy[i, 0] + text_w - 1
        text_y2 = detections.xyxy[i, 1] + text_h - 1

        draw.rectangle(
            (
                detections.xyxy[i, 0].item(),
                detections.xyxy[i, 1].item(),
                text_x2.item(),
                text_y2.item(),
            ),
            fill=color,
        )
        draw.text(
            (detections.xyxy[i, 0].item(), detections.xyxy[i, 1].item()),
            caption,
            fill="black",
            font=font,
        )

    return image


def draw_boxes(image, boxes, score=None, color=(255, 255, 255)):
    """
    Args:
        img:
            PIL type image
        boxes:
            xyxy

    Returns:
        pil type image:
            アノテーションを追加した画像
    """

    # フォントを作成する。
    fontsize = max(10, int(0.03 * min(image.size)))
    fontname = "DejaVuSerif-Bold"
    font = ImageFont.truetype(fontname, size=fontsize)

    draw = ImageDraw.Draw(image, mode="RGBA")

    for i in range(len(boxes)):
        # 矩形を描画する
        draw.rectangle(boxes[i].tolist(), width=5, outline=color)

        if score is not None:
            # ラベルを描画する。
            caption = str(round(score[i].item() * 100)) + "%"
            text_w, text_h = draw.textsize(caption, font=font)
            text_x2 = boxes[i][0] + text_w - 1
            text_y2 = boxes[i][1] + text_h - 1
            draw.rectangle((boxes[i][0], boxes[i][1], text_x2, text_y2), fill=color)
            draw.text((boxes[i][0], boxes[i][1]), caption, fill="black", font=font)

    return image
