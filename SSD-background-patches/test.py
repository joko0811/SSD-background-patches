import torch
import cv2

from torchvision.datasets.coco import CocoDetection

from util.img import pil2cv
from util import box
from model import yolo
import proposed_func as pf


def test_loss():
    train_path = "./coco2014/images/train2014/"
    train_annfile_path = "./coco2014/annotations/instances_train2014.json"
    coco_train = CocoDetection(root=train_path,
                               annFile=train_annfile_path)
    img, _ = coco_train[0]
    img = pil2cv(img)

    if torch.cuda.is_available():
        dtype = torch.float
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        gpu_img = torch.tensor(
            img, device=device, dtype=dtype, requires_grad=True)


def main():
    test_loss()


if __name__ == '__main__':
    main()
