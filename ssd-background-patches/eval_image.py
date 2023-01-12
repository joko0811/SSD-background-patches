
import os
import sys

import torch
from torchvision import transforms

from tqdm import tqdm
from tensorboardX import SummaryWriter

from model import yolo, yolo_util
from dataset.coco import load_class_names
from dataset.simple import DirectoryImageDataset
from dataset.detection import DirectoryImageWithDetectionDataset
from box import boxio
from imageutil import imgdraw


def generate_data(path):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    out_path = os.path.join(path, "box/")
    image_dir_path = os.path.join(path, "image")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # class_names = load_class_names("./coco2014/coco.names")

    yolo_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((416, 416)),
    ])

    image_set = DirectoryImageDataset(
        image_dir_path, transform=yolo_transforms)
    image_loader = torch.utils.data.DataLoader(image_set)

    model = yolo.load_model(
        "weights/yolov3.cfg",
        "weights/yolov3.weights")
    model.eval()

    for (image, image_path) in tqdm(image_loader):

        if torch.cuda.is_available():
            gpu_image = image.to(
                device=device, dtype=torch.float)

        output = model(gpu_image)
        nms_out = yolo_util.nms(output)
        detections = yolo_util.detections_yolo(nms_out[0])

        det_str = boxio.format_yolo(detections, image.shape[-2:])

        image_name = os.path.basename(image_path[0]).split(".")[0]
        det_file_path = os.path.join(out_path, image_name+".txt")

        with open(det_file_path, "w") as f:
            f.write(det_str)

    return


def tbx_monitor(path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    box_path = os.path.join(path, "box/")
    image_dir_path = os.path.join(path, "image")

    tbx_writer = SummaryWriter("outputs/tbx/")

    class_names = load_class_names("./datasets/coco2014/coco.names")

    yolo_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((416, 416)),
    ])

    image_set = DirectoryImageWithDetectionDataset(
        image_dir_path, box_path, transform=yolo_transforms)
    image_loader = torch.utils.data.DataLoader(image_set)

    for i, (image, image_path, label_list, xywh_list) in enumerate(tqdm(image_loader)):
        detection = boxio.detections_base(
            torch.tensor(label_list), torch.tensor(xywh_list))
        anno_image = imgdraw.draw_annotations(
            image[0], detection, class_names, in_confidences=False)
        tbx_writer.add_image("annotation_image", anno_image, i)

    return


def main():
    # path = sys.argv[1]
    path = "datasets/gait_test1/"
    generate_data(path)
    tbx_monitor(path)


if __name__ == "__main__":
    main()
