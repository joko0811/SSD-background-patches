import time

import numpy as np
import torch
from torchvision import transforms, ops

from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
from pytorchyolo.utils import utils

from .yolo import load_model


def image_setup(img):
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(416)])(
            (img, np.zeros((1, 5))))[0].unsqueeze(0)
    return input_img


def detect(img):
    model = load_model(
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
    model = load_model(
        "weights/yolov3.cfg",
        "weights/yolov3.weights")
    model.train()
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
        above_threshold_x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not above_threshold_x.shape[0]:
            continue

        # Compute conf
        # conf = obj_conf * cls_conf
        confidences = above_threshold_x[:, 5:] * above_threshold_x[:, 4:5]

        # Box (center x, center y, width, height) to (left, top, right, bottom)
        box = utils.xywh2xyxy(above_threshold_x[:, :4])

        # Detections matrix nx6 (xyxy, cls_conf, cls_idx) (old)
        # Detections matrix nx6 (xywh,xyxy,cls_score,cls_idx,cls_scores)
        if multi_label:
            i, j = (confidences > conf_thres).nonzero(as_tuple=False).T
            reshaped_x = torch.cat(
                (above_threshold_x[i, :4], box[i], confidences[i, j, None], j[:, None].float(), confidences[i]), 1)
        else:  # best class only
            conf, j = confidences.max(1, keepdim=True)
            reshaped_x = torch.cat((x[:, :4], box, conf, j.float(), confidences), 1)[
                conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            filtered_x = reshaped_x[(reshaped_x[:, 9:10] == torch.tensor(
                classes, device=reshaped_x.device)).any(1)]
        else:
            filtered_x = reshaped_x

        # Check shape
        n = filtered_x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            sorted_x = filtered_x[filtered_x[:, 8].argsort(descending=True)[
                :max_nms]]
        else:
            sorted_x = filtered_x

        # Batched NMS
        c = sorted_x[:, 9:10] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = sorted_x[:, 4:8] + c, sorted_x[:, 8]
        nms_idx = ops.nms(boxes, scores, iou_thres)  # NMS
        if nms_idx.shape[0] > max_det:  # limit detections
            below_upper_limit_nms_idx = nms_idx[:max_det]
        else:
            below_upper_limit_nms_idx = nms_idx

        output[xi] = sorted_x[below_upper_limit_nms_idx]

        if (time.time() - t) > time_limit:
            print(
                f'WARNING: NMS time limit {time_limit}s eclass_extract_xceeded')
            break  # time limit eclass_extract_xceeded

    return output


class detections_base:
    # yolo_out=[x,y,w,h,confidence_score,class_scores...]
    # nms_out=[x1,y1,x2,y2,x,y,w,h,confidence_score,class_scores...]
    def __init__(self, data, is_nms=True):
        self.data = data
        self.total_det = len(self.data)
        self.xywh = self.data[:, :4]
        self.is_mns = is_nms
        if is_nms:
            self.xyxy = self.data[:, 4:8]
            self.confidences = self.data[:, 8]
            self.class_labels = self.data[:, 9].to(torch.int64)
            self.class_scores = self.data[:, 10:]
        else:
            self.xyxy = utils.xywh2xyxy(self.xywh)
            self.confidences = self.data[:, 4]
            self.class_scores = self.data[:, 5:]
            self.class_labels = self.class_scores.argmax(dim=1).to(torch.int64)


class detections_ground_truth(detections_base):
    def set_group_info(self, labels):
        self.group_labels = labels
        self.total_group = int(self.group_labels.max().item())+1


class detections_loss(detections_base):
    def set_loss_info(self, nearest_gt_idx, z, r):
        self.nearest_gt_idx = nearest_gt_idx
        self.z = z
        self.r = r
