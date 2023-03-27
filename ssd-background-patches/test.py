import hydra
import torch
from PIL import Image
from imageutil import imgdraw
from torchvision import transforms

from model.S3FD import s3fd
from model.s3fd_util import S3fdResize


def s3fd_demo():

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    S3FD_FACE_TRANSFORMS = transforms.Compose([
        S3fdResize(),
        transforms.PILToTensor(),
    ])

    image = Image.open("data/test.jpg")
    # 変換前の画像の座標に変換するための配列
    scale = torch.tensor(
        [image.width, image.height, image.width, image.height]).to(device=device, dtype=torch.float)

    # transform
    s3fd_image = (S3FD_FACE_TRANSFORMS(image)).to(
        device=device, dtype=torch.float).unsqueeze(0)

    def _load_model(path):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # Select device for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = s3fd.build_s3fd('test', 2)
        model.load_state_dict(torch.load(path))

        return model.to(device)
    model = _load_model("weights/s3fd.pth")

    model.eval()

    thresh = 0.6
    with torch.no_grad():
        output = model(s3fd_image)
        extract_output = output[output[..., 0] >= thresh]

        score = extract_output[..., 0]
        box = extract_output[..., 1:]*scale
        anno_image = imgdraw.draw_boxes(image, box)

    anno_image.save("hoge.png")


@ hydra.main(version_base=None, config_path="../conf/dataset/", config_name="casia_train")
def hydra_test(cfg):
    return


def main():
    # hydra_test()
    s3fd_demo()


if __name__ == '__main__':
    main()
