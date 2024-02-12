from abc import ABCMeta, abstractclassmethod
from torchvision.transforms import Compose


class BaseTrainer(metaclass=ABCMeta):
    @abstractclassmethod
    def get_dataloader(self):
        pass

    @abstractclassmethod
    def load_model(self):
        pass

    @abstractclassmethod
    def make_detections_list(self):
        pass


class BackgroundBaseTrainer(BaseTrainer):
    def transformed2pil(self):
        pass
