import os
import argparse

import torch
from torchvision import transforms
from torchvision.datasets.coco import CocoDetection

from tqdm import tqdm

from model import yolo, yolo_util
from dataset.coco import load_class_names
from dataset.simple import DirectoryDataset
from evaluation.detection import list_iou, calc_class_TP, calc_class_FP, calc_class_FN


def format_detections(detections, det_idx, label_names):
    label = label_names[detections.class_labels[det_idx]]
    conf = detections.confidences[det_idx]
    box = detections.xyxy[det_idx]

    return label+" "+str(conf.item())+" "+str(int(box[0].item()))+" "+str(int(box[1].item()))+" "+str(int(box[2].item()))+" "+str(int(box[3].item()))


def format_yolo(detections, det_idx, label_names):
    label_idx = detections.class_labels[det_idx]
    box = detections.xywh[det_idx]

    return str(label_idx.item())+" "+str(int(box[0].item()))+" "+str(int(box[1].item()))+" "+str(int(box[2].item()))+" "+str(int(box[3].item()))


def save_detections(model_out, class_names, image_path, format):
    nms_out = yolo_util.nms(model_out)

    detections = yolo_util.detections_base(nms_out[0])
    det_str = ""

    for i in range(detections.total_det):
        det_str += format(
            detections, i, class_names)+"\n"

    with open(image_path, mode='w') as f:
        f.write(det_str)


def evaluation(path):

    out_path = path+"box/"
    gt_out_path = out_path+"groundtruths/"
    dt_out_path = out_path+"detections/"

    if not os.path.exists(dt_out_path):
        os.makedirs(out_path)
        os.makedirs(gt_out_path)
        os.makedirs(dt_out_path)

    class_names = load_class_names("./coco2014/coco.names")
    gt_image_path = "./coco2014/images/train2014/"
    gt_annfile_path = "./coco2014/annotations/instances_train2014.json"

    yolo_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((416, 416)),
    ])

    gt_set = CocoDetection(root=gt_image_path, annFile=gt_annfile_path,
                           transform=yolo_transforms)
    gt_loader = torch.utils.data.DataLoader(gt_set)

    adv_set = DirectoryDataset(
        image_path=path, transform=yolo_transforms)
    adv_loader = torch.utils.data.DataLoader(adv_set)

    set_size = len(adv_set)

    model = yolo.load_model(
        "weights/yolov3.cfg",
        "weights/yolov3.weights")
    model.eval()

    for (gt_image, _), (adv_image, adv_image_path) in tqdm(zip(gt_loader, adv_loader), total=set_size):

        if torch.cuda.is_available():
            gt_image = gt_image.to(
                device='cuda:0', dtype=torch.float)
            adv_image = adv_image.to(
                device='cuda:0', dtype=torch.float)

        image_name = adv_image_path[0].split(os.sep)[-1].split(".")[0]

        gt_output = model(gt_image)
        save_detections(gt_output, class_names, gt_out_path +
                        image_name+".txt", format_yolo)

        dt_output = model(adv_image)
        save_detections(dt_output, class_names, dt_out_path +
                        image_name+".txt", format_detections)

    return


def main():
    arg_parser = argparse.ArgumentParser(
        description="evaluate adversarial image")
    arg_parser.add_argument(
        "path", type=str, help="select adversarial image path")
    args = arg_parser.parse_args()
    path = args.path
    evaluation(path)


if __name__ == "__main__":
    main()
