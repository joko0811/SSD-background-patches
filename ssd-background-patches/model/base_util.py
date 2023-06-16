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

    # casia gait b dataset画像のの変形後のサイズ(HW)を返す
    # TODO: 動的に取得するやり方を考える
    def get_image_size(self):
        pass
