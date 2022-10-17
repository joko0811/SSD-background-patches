import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.coco import CocoDetection

from pytorchyolo import detect, models
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS

from util import pil2cv


def test():

    coco_train = CocoDetection(root="./coco2014/images/train2014/",
                               annFile="./coco2014/annotations/instances_train2014.json")

    img, target = coco_train[0]
    img = pil2cv(img)

    model = models.load_model(
        "weights/yolov3.cfg",
        "weights/yolov3.weights")

    model.eval()  # Set model to evaluation mode

    # Configure input
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(416)])(
            (img, np.zeros((1, 5))))[0].unsqueeze(0)

    if torch.cuda.is_available():
        input_img = input_img.to("cuda")

    # Get detections
    with torch.no_grad():
        yolo_out = model(input_img)
        class_num = yolo_out.shape[2]-5
        class_score = yolo_out[2]
        print(class_score)


if __name__ == "__main__":
    test()
