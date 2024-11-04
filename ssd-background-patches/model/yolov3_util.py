import torch
from torchvision import transforms
from omegaconf import DictConfig
from torch.utils.data import Subset, DataLoader

from model.yolov3.models.yolo import DetectMultiBackend
from model.yolov3.utils.general import non_max_suppression
from model.base_util import BaseTrainer
from detection.detection_base import ObjectDetectionBase


class YoloTrainer(BaseTrainer):

    YOLO_TRANSFORMS = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((416, 416)),
        ]
    )

    def __init__(self, model_conf: DictConfig, dataset_factory):
        self.model_conf = model_conf
        dataset = Subset(
            dataset_factory(transform=self.YOLO_TRANSFORMS),
            [i for i in range(3000)],
        )
        self.dataloader = DataLoader(dataset, batch_size=1)

    def get_transform(self):
        return self.YOLO_TRANSFORMS

    def get_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader

    def load_model(self, device=None, mode="test"):
        model = load_model(weights_path=self.model_conf.weights_path, device=device)
        # model.to(device=device)
        return model

    def make_detections_list(self, data_list, thresh=0.6, scale=None, img_size=None):
        detections_list = list()
        class_number = 80  # coco
        pred = non_max_suppression(data_list)

        for det in pred:
            if det.nelement() == 0:
                detections_list.append(None)
            else:
                detections_list.append(
                    ObjectDetectionBase(
                        det[:, 4], det[:, :4], det[:, 5], det[:, 6:], is_xywh=False
                    ),
                )

        return detections_list


def load_model(weights_path, device):
    # model = Model(weights_path)
    model = DetectMultiBackend(weights_path, device)
    return model
