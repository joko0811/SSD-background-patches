import torch
from box import boxconv


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


def format_detections(detections, label_names):
    det_str = ""

    for det_idx in range(detections.total_det):
        label = label_names[detections.class_labels[det_idx]]
        conf = detections.confidences[det_idx]
        box = detections.xyxy[det_idx]

        det_str += (
            label + " " +
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

    label_list = list()
    xyxy_list = list()

    for det_str in det_str_list:
        det_info = det_str.split()

        label = det_info[0]
        x1 = det_info[1]
        y1 = det_info[2]
        x2 = det_info[3]
        y2 = det_info[4]

        label_list.append(label)
        xyxy_list.append([x1, y1, x2, y2])

    return label_list, xyxy_list


def format_yolo(detections, image_hw):
    det_str = ""

    for det_idx in range(detections.total_det):
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


class detections_base:
    def __init__(self, label_list, box_list, is_xywh=True):
        self.class_labels = label_list.to(torch.int64)
        self.total_det = len(self.class_labels)
        if is_xywh:
            self.xywh = box_list
            self.xyxy = boxconv.xywh2xyxy(self.xywh)
        else:
            self.xyxy = box_list
            self.xywh = boxconv.xyxy2xywh(self.xyxy)
