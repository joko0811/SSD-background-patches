import numpy as np

import torch
from torchvision import transforms

from model.S3FD import s3fd

from model.base_util import BackgroundBaseTrainer
from detection.detection_base import DetectionsBase
from omegaconf import DictConfig


class S3fdResize:
    def __init__(self, scale_reference_size=(1700, 1200)):
        self.scale_reference = scale_reference_size[0] * scale_reference_size[1]

    def __call__(self, pic):
        width, height = transforms.functional.get_image_size(pic)
        x_area = width * height

        scale = np.sqrt(self.scale_reference / x_area)
        out_size = (round(height * scale), round(width * scale))
        return transforms.functional.resize(img=pic, size=out_size)


class S3fdDeTransform:
    """S3fdTrainer.S3FD_TRANSFORMSの逆変換"""

    def __init__(self):
        pass

    def __call__(self, pic, scale=None):
        if scale is None:
            return transforms.functional.to_pil_image(pic / 255)
        rs_pic = transforms.functional.resize(pic, scale)
        return transforms.functional.to_pil_image(rs_pic / 255)


class S3fdTrainer(BackgroundBaseTrainer):

    S3FD_TRANSFORMS = transforms.Compose(
        [
            S3fdResize(),
            transforms.PILToTensor(),
        ]
    )

    def __init__(self, model_conf: DictConfig, dataset_factory):
        self.model_conf = model_conf
        dataset = dataset_factory(transform=self.S3FD_TRANSFORMS)
        self.dataloader = torch.utils.data.DataLoader(dataset)

    def get_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader

    def load_model(self, device=None, mode="test"):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        torch.set_default_device(device)

        model = s3fd.build_s3fd(mode, 2)
        model.load_state_dict(torch.load(self.model_conf.weight_path))

        return model.to(device)

    def make_detections_list(self, data_list, thresh=0.6, scale=None, img_size=None):
        detections_list = list()
        for data in data_list:
            extract_data = data[data[..., 0] >= thresh]
            if extract_data.nelement() != 0:
                detections_list.append(
                    DetectionsBase(
                        extract_data[..., 0], extract_data[..., 1:], is_xywh=False
                    )
                )
                # detection_class(extract_data, scale_list[i]))
            else:
                detections_list.append(None)

        return detections_list

    def transformed2pil(self, pic, scale=None):
        desize = S3fdDeTransform()
        return desize(pic, scale)
