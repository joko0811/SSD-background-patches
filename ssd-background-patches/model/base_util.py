from abc import ABCMeta, abstractclassmethod
from torchvision.transforms import Compose


class BaseTrainer(metaclass=ABCMeta):

    @abstractclassmethod
    def get_dataloader(self, *args, **kwargs) -> Compose:
        pass

    @abstractclassmethod
    def load_model(self, *args, **kwargs):
        pass

    @abstractclassmethod
    def make_detections_list(self, *args, **kwargs):
        pass


class BackgroundBaseTrainer(BaseTrainer):
    def transformed2pil():
        pass
