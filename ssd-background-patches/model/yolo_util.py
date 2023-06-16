import time

import torch
from torchvision import transforms, ops

from pytorchyolo.utils import utils

from model import yolo

from model.base_util import BaseTrainer
from detection.detection_base import ObjectDetectionBase
import hydra
from omegaconf import DictConfig


class YoloTrainer(BaseTrainer):

    YOLO_TRANSFORMS = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((416, 416)),
        ]
    )

    def __init__(self, model_conf: DictConfig, dataset_conf: DictConfig):
        self.model_conf = model_conf
        dataset = hydra.utils.instantiate(dataset_conf, transforms=self.YOLO_TRANSFORMS)
        self.dataloader = torch.utils.data.DataLoader(dataset)

    def get_transform(self):
        return self.YOLO_TRANSFORMS

    def get_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader

    def load_model(self):
        return yolo.load_model(
            model_path=self.model_conf.model_path,
            weights_path=self.model_conf.weights_path,
        )

    def make_detections_list(self, data_list):
        detections_list = list()
        for data in data_list:
            if data.nelement() != 0:
                detections_list.append(
                    ObjectDetectionBase(
                        data[:, 8], data[:, 4:8], data[:, 10:], is_xywh=False
                    )
                )
                # detection_class(data, is_nms))
            else:
                detections_list.append(None)

        return detections_list


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
        output = [torch.zeros((0, 10 + nc), device="cuda")] * prediction.shape[0]
    else:
        output = [torch.zeros((0, 10 + nc), device="cpu")] * prediction.shape[0]

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
                (
                    above_threshold_x[i, :4],
                    box[i],
                    confidences[i, j, None],
                    j[:, None].float(),
                    confidences[i],
                ),
                1,
            )
        else:  # best class only
            conf, j = confidences.max(1, keepdim=True)
            reshaped_x = torch.cat((x[:, :4], box, conf, j.float(), confidences), 1)[
                conf.view(-1) > conf_thres
            ]

        # Filter by class
        if classes is not None:
            filtered_x = reshaped_x[
                (
                    reshaped_x[:, 9:10]
                    == torch.tensor(classes, device=reshaped_x.device)
                ).any(1)
            ]
        else:
            filtered_x = reshaped_x

        # Check shape
        n = filtered_x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            sorted_x = filtered_x[filtered_x[:, 8].argsort(descending=True)[:max_nms]]
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
            print(f"WARNING: NMS time limit {time_limit}s eclass_extract_xceeded")
            break  # time limit eclass_extract_xceeded

    return output
