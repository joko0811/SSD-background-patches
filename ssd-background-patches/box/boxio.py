import torch
import glob

from detection.detection_base import DetectionsBase, ObjectDetectionBase


def generate_integrated_xyxy_list(path, max_iter=None):
    path_list = sorted(glob.glob("%s/*.*" % path))
    if max_iter is not None:
        path_list = path_list[:max_iter]

    conf_list = list()
    xyxy_list = list()

    for path in path_list:
        parse_tuple = parse_detections(path)
        if parse_tuple is not None:
            conf_l, xyxy_l = parse_tuple
            for conf, xyxy in zip(conf_l, xyxy_l):
                conf_list.append(conf)
                xyxy_list.append(xyxy.unsqueeze(0))

    return torch.tensor(conf_list).contiguous(), torch.cat(xyxy_list).contiguous()


def format_boxes(boxes, **kargs):
    box_str = ""

    for box_idx in range(boxes):
        x = boxes[box_idx, 0]
        y = boxes[box_idx, 0]
        w = boxes[box_idx, 0]
        h = boxes[box_idx, 0]
        box_str += (
            str(x.item()) + " " +
            str(y.item()) + " " +
            str(w.item()) + " " +
            str(h.item()) + "\n"
        )

    return box_str


def format_detections(detections: DetectionsBase):
    det_str = ""

    for det_idx in range(len(detections)):
        conf = detections.conf[det_idx]
        box = detections.xyxy[det_idx]

        det_str += (
            str(conf.item()) + " " +
            str(box[0].item()) + " " +
            str(box[1].item()) + " " +
            str(box[2].item()) + " " +
            str(box[3].item()) + "\n"
        )

    return det_str


def parse_detections(path):
    with open(path, "r") as f:
        det_str_list = f.readlines()

    if len(det_str_list) == 0:
        return None

    conf_list = list()
    xyxy_list = list()

    for det_str in det_str_list:
        det_info = det_str.split()

        conf = float(det_info[0])
        x1 = float(det_info[1])
        y1 = float(det_info[2])
        x2 = float(det_info[3])
        y2 = float(det_info[4])

        conf_list.append(torch.tensor(conf))
        xyxy_list.append(torch.tensor([x1, y1, x2, y2]).unsqueeze(0))

    return torch.tensor(conf_list).contiguous(), torch.cat(xyxy_list).contiguous()


def format_yolo(detections: ObjectDetectionBase, image_hw):
    det_str = ""

    for det_idx in range(len(detections)):
        label_idx = detections.class_labels[det_idx]
        box = detections.xywh[det_idx]
        yolo_x = box[0].item()/image_hw[1]
        yolo_y = box[1].item()/image_hw[0]
        yolo_w = box[2].item()/image_hw[1]
        yolo_h = box[3].item()/image_hw[0]

        det_str += (
            str(label_idx.item()) + " " +
            str(yolo_x) + " " +
            str(yolo_y) + " " +
            str(yolo_w) + " " +
            str(yolo_h) + "\n"
        )

    return det_str


def parse_yolo(path, image_hw):

    with open(path, "r") as f:
        det_str_list = f.readlines()

    label_list = list()
    xywh_list = list()

    for det_str in det_str_list:
        det_info = det_str.split()

        label = int(det_info[0])
        x = float(det_info[1])*image_hw[1]
        y = float(det_info[2])*image_hw[0]
        w = float(det_info[3])*image_hw[1]
        h = float(det_info[4])*image_hw[0]

        label_list.append(label)
        xywh_list.append([x, y, w, h])

    return label_list, xywh_list
