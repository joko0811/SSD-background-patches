import torch
from PIL import Image
from model import s3fd_util
from imageutil import imgdraw


def s3fd_demo():

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    image = Image.open("data/test.jpg")
    # 変換前の画像の座標に変換するための配列
    scale = torch.tensor(
        [image.width, image.height, image.width, image.height]).to(device=device, dtype=torch.float)

    # transform
    s3fd_image = (s3fd_util.S3FD_TRANSFORMS(
        image)-s3fd_util.S3FD_NOMALIZED_ARRAY).to(device=device, dtype=torch.float).unsqueeze(0)

    thresh = 0.6
    model = s3fd_util.load_model("weights/s3fd.pth")

    model.eval()

    with torch.no_grad():
        output = model(s3fd_image)
        extract_output = output[output[..., 0] >= thresh]

        score = extract_output[..., 0]
        box = extract_output[..., 1:]*scale
        anno_image = imgdraw.draw_boxes(image, box)

    anno_image.save("hoge.png")


def main():
    s3fd_demo()


if __name__ == '__main__':
    main()
