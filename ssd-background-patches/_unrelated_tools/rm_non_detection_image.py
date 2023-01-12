import os

from tqdm import tqdm

import torch

from model import yolo, yolo_util
from dataset.simple import DirectoryImageDataset


def main():

    print(os.getcwd())

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    setting_path = "weights/yolov3.cfg"
    weight_path = "weights/yolov3.weights"
    model = yolo.load_model(setting_path, weight_path)

    model.eval()

    image_set_path = "datasets/casiagait_b_video90/image/"
    image_set = DirectoryImageDataset(
        image_set_path, transform=yolo_util.YOLO_TRANSFORMS)

    image_loader = torch.utils.data.DataLoader(image_set)

    for (image, image_path) in tqdm(image_loader):
        gpu_image = image.to(device=device, dtype=torch.float)
        gt_output = model(gpu_image)
        gt_nms_out = yolo_util.nms(gt_output)
        gt_detections = yolo_util.detections_yolo(gt_nms_out[0])

        if gt_detections.total_det == 0:
            os.remove(image_path[0])


if __name__ == "__main__":
    main()
