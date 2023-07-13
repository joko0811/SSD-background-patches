import cv2


def video2image_list(path: str, per_frame: int = 1):
    """動画を画像のリストに変換する
    Args:
        path: 動画ファイルのパス
        per_frame: 何フレーム毎に画像を保存するかを指定する
    Return:
        image_list: cv outputarrayのリスト
    """
    vidcap: cv2.VideoCapture = cv2.VideoCapture(path)
    current_frame: int = 0
    image_list: list = list()

    while True:
        success, image = vidcap.read()

        if success:
            current_frame += 1
            if current_frame % per_frame == 0:
                image_list.append(image)
        else:
            break

    return image_list
