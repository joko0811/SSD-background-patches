import torch
import numpy as np

from torchvision import transforms
from torchvision.datasets.coco import CocoDetection

from pytorchyolo import models
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS

from util.img import pil2cv
from util.detection import nms


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
        # yolo_out=[center_x,center_y,w,h,confidence_score,class_scores...]
        yolo_out = model(input_img)

        # yolo_out=[left_x,top_y,right_x,bottom_y,class_scores...]
        yolo_out = nms(yolo_out)
        print(yolo_out)


if __name__ == "__main__":
    test()
