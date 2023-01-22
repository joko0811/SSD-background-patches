import torch
from PIL import Image
from model import s3fd_util
from torchvision import transforms
from imageutil import imgdraw


def s3fd_demo():

    thresh = 0.6

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    image = Image.open("data/test.jpg")
    gpu_image = transforms.functional.to_tensor(
        image).to(device=device, dtype=torch.float)

    model = s3fd_util.load_model("weights/s3fd.pth")

    s3fd_image, scale = s3fd_util.image_encode(image)
    s3fd_image = s3fd_image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(s3fd_image)
        extract_output = output[output[..., 0] >= thresh]
        box = extract_output[..., 1:]*scale
        score = extract_output[..., 0]
        anno_image = imgdraw.draw_boxes(gpu_image, box)

    transforms.functional.to_pil_image(anno_image).save("hoge.png")
    return


def test_ende():
    in_image = Image.open(
        "datasets/casiagait_b_video90/image/001-bg-01-090-036.png")
    data, scale = s3fd_util.image_encode(in_image)
    out_image = s3fd_util.image_decode(data[0], scale)
    out_image.save("out.jpg")


def main():
    s3fd_demo()


if __name__ == '__main__':
    main()
