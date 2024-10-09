import torch
from torchvision import transforms

from model.MTCNN.model import MTCNN

from model.base_util import BackgroundBaseTrainer
from detection.detection_base import DetectionsBase
from omegaconf import DictConfig


class MTCNNTrainer(BackgroundBaseTrainer):

    MTCNN_TRANSFORMS = transforms.Compose([transforms.PILToTensor()])

    def __init__(self, model_conf: DictConfig, dataset_factory):
        self.model_conf = model_conf
        dataset = dataset_factory(transform=self.MTCNN_TRANSFORMS)
        self.dataloader = torch.utils.data.DataLoader(dataset)

    def get_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader

    def load_model(self, device=None, mode="test"):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MTCNN(keep_all=True, device=device)

        return model

    def make_detections_list(self, data_list, thresh=0.6, scale=None, img_size=None):
        detections_list = list()
        boxes, probs = data_list
        for box, prob, s in zip(boxes, probs, scale):
            if box.nelement() != 0:
                detections_list.append(DetectionsBase(prob, box / s, is_xywh=False))
            else:
                detections_list.append(None)
        return detections_list

    def transformed2pil(self, pic, scale=None):
        if scale is None:
            return transforms.functional.to_pil_image(pic / 255)
        rs_pic = transforms.functional.resize(pic, scale)
        return transforms.functional.to_pil_image(rs_pic / 255)
