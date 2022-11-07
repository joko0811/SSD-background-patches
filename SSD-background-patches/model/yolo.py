import time

import numpy as np
import torch
import torchvision

from pytorchyolo import models
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
from pytorchyolo.utils import utils


def image_setup(img):
    input_img = torchvision.transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(416)])(
            (img, np.zeros((1, 5))))[0].unsqueeze(0)
    return input_img


def detect(img):
    model = models.load_model(
        "weights/yolov3.cfg",
        "weights/yolov3.weights")

    model.eval()  # Set model to evaluation mode

    # Get detections
    with torch.no_grad():
        # yolo_out=[center_x,center_y,w,h,confidence_score,class_scores...]
        yolo_out = model(img)

        # yolo_out=[left_x,top_y,right_x,bottom_y,class_scores...]
        return yolo_out


def detect_with_grad(img):
    model = models.load_model(
        "weights/yolov3.cfg",
        "weights/yolov3.weights")
    model.eval()  # Set model to evaluation mode
    yolo_out = model(img)
    return yolo_out


def nms(prediction, conf_thres=0.25, iou_thres=0.45, classes=None):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
        detections with shape: nx6 (x1, y1, x2, y2, conf, cls...)
    """

    nc = prediction.shape[2] - 5  # number of classes

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()

    if torch.cuda.is_available():
        output = [torch.zeros((0, 10+nc), device="cuda")] * prediction.shape[0]
    else:
        output = [torch.zeros((0, 10+nc), device="cpu")] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (left, top, right, bottom)
        box = utils.xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, cls_conf, cls_idx) (old)
        # Detections matrix nx6 (xywh,xyxy,cls_score,cls_idx,cls_scores)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            class_extract_x = torch.cat(
                (x[i, :4], box[i], x[i, j + 5, None], j[:, None].float(), x[i, 5:]), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            class_extract_x = torch.cat((x[:, :4], box, conf, j.float(), x[:, 5:]), 1)[
                conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            class_extract_x = class_extract_x[(class_extract_x[:, 9:10] == torch.tensor(
                classes, device=class_extract_x.device)).any(1)]

        # Check shape
        n = class_extract_x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            class_extract_x = class_extract_x[class_extract_x[:, 8].argsort(descending=True)[
                :max_nms]]

        # Batched NMS
        c = class_extract_x[:, 9:10] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = class_extract_x[:, 4:8] + c, class_extract_x[:, 8]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = class_extract_x[i]

        if (time.time() - t) > time_limit:
            print(
                f'WARNING: NMS time limit {time_limit}s eclass_extract_xceeded')
            break  # time limit eclass_extract_xceeded

    return output


class detections_base:
    # yolo_out=[x,y,w,h,confidence_score,class_scores...]
    # nms_out=[x1,y1,x2,y2,x,y,w,h,confidence_score,class_scores...]
    def __init__(self, data):
        self.data = data
        self.xywh = self.data[:, :4]
        self.xyxy = utils.xywh2xyxy(self.xywh)
        self.conf = self.data[:, 4]
        self.class_scores = self.data[:, 5:]
        self.class_labels = self.class_scores.argmax(dim=1).to(torch.int64)
        self.total_det = len(self.data)


class detections_nms_out(detections_base):
    # yolo_out=[x,y,w,h,confidence_score,class_scores...]
    # nms_out=[x1,y1,x2,y2,x,y,w,h,confidence_score,class_scores...]
    # Detections matrix nx6 (xywh,xyxy,cls_score,cls_idx,cls_scores)
    def __init__(self, data):
        self.data = data
        self.xywh = self.data[:, :4]
        self.xyxy = self.data[:, 4:8]
        self.confidences = self.data[:, 8]
        self.class_labels = self.data[:, 9].to(torch.int64)
        self.class_scores = self.data[:, 10:]
        self.total_det = len(self.data)


class detections_loss(detections_base):
    def set_loss_info(self, nearest_gt_idx, z, r):
        self.nearest_gt_idx = nearest_gt_idx
        self.z = z
        self.r = r
