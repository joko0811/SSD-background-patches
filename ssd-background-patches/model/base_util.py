from abc import ABCMeta, abstractclassmethod
from torchvision.transforms import Compose


class BaseUtilizer(metaclass=ABCMeta):
    @abstractclassmethod
    def get_transform(self, *args, **kwargs) -> Compose:
        pass

    @abstractclassmethod
    def load_model(self, *args, **kwargs):
        pass

    @abstractclassmethod
    def make_detections_list(self, *args, **kwargs):
        pass
