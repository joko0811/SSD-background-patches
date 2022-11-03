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
    img_input = image_setup(img)
    model = models.load_model(
        "weights/yolov3.cfg",
        "weights/yolov3.weights")

    model.eval()  # Set model to evaluation mode

    # Get detections
    with torch.no_grad():
        # yolo_out=[center_x,center_y,w,h,confidence_score,class_scores...]
        yolo_out = model(img_input)

        # yolo_out=[left_x,top_y,right_x,bottom_y,class_scores...]
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
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

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

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            class_extract_x = torch.cat(
                (box[i], x[i, j + 5, None], j[:, None].float()), 1)
            # class_extract_x=[[x,y,x2,y2,class_score,class_index],...]
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            class_extract_x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(
                classes, device=class_extract_x.device)).any(1)]
            class_extract_x = class_extract_x[(class_extract_x[:, 5:6] == torch.tensor(
                classes, device=class_extract_x.device)).any(1)]

        # Check shape
        n = class_extract_x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            class_extract_x = class_extract_x[class_extract_x[:, 4].argsort(descending=True)[
                :max_nms]]

        # Batched NMS
        c = class_extract_x[:, 5:6] * max_wh  # classes
        # boclass_extract_xes (offset by class), scores
        boxes, scores = class_extract_x[:, :4] + c, class_extract_x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        x = x[i]
        class_extract_x = class_extract_x[i]

        output[xi] = torch.cat(
            (class_extract_x[:, :4], x[:, 5:]), 1).detach().cpu()

        if (time.time() - t) > time_limit:
            print(
                f'WARNING: NMS time limit {time_limit}s eclass_extract_xceeded')
            break  # time limit eclass_extract_xceeded

    return output


class detections:
    def __init__(self, img):
        yolo_out = detect(img)
        self.detections = nms(yolo_out)
        self.boxes = self.detections[:, :4]  # x1y1x2y2
        self.conf = self.detections[:, 4]
        self.class_label = np.argmax(self.detections, axis=1)
        self.class_scores = self.detections[:, 4:]
