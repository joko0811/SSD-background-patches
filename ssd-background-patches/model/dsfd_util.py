import math

import torch
from torchvision import transforms

from omegaconf import DictConfig

from detection.detection_base import DetectionsBase
from model.base_util import BackgroundBaseTrainer
from model.DSFD.conf.widerface import widerface_640
from model.DSFD.model.face_ssd import build_ssd


class DsfdResize:
    def __init__(self, scale_reference_size=(2000, 2000)):
        self.scale_reference = scale_reference_size[0]*scale_reference_size[1]

    def __call__(self, pic):
        width, height = transforms.functional.get_image_size(pic)
        x_area = width*height

        shrink = max(1, math.sqrt(self.scale_reference/x_area))
        out_size = (round(height*shrink), round(width*shrink))
        return transforms.functional.resize(img=pic, size=out_size)


class DsfdTrainer(BackgroundBaseTrainer):
    # casia gait b dataset画像のの変形後のサイズ(HW)
    # TODO: 動的に取得するやり方を考える
    image_size = (840, 840)
    DSFD_TRANSFORMS = transforms.Compose([
        DsfdResize(),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((104, 117, 123), 1),
    ])

    def __init__(self, model_conf: DictConfig, dataset_factory):
        self.model_conf = model_conf
        dataset = dataset_factory(transform=self.DSFD_TRANSFORMS)
        self.dataloader = torch.utils.data.DataLoader(dataset)

    def get_dataloader(self):
        return self.dataloader

    def load_model(self):
        cfg = widerface_640
        net = build_ssd('test', cfg['min_dim'])  # initialize SSD
        net.load_state_dict(torch.load('weights/WIDERFace_DSFD_RES152.pth'))
        return net

    def make_detections_list(self, data_list, thresh=0.01):
        detections_list = list()
        for data in data_list:
            extract_data = data[data[..., 0] >= thresh]
            if extract_data.nelement() != 0:
                detections_list.append(
                    DetectionsBase(
                        extract_data[..., 0], extract_data[..., 1:], is_xywh=False))
                # detection_class(extract_data, scale_list[i]))
            else:
                detections_list.append(None)

        return detections_list

    def transformed2pil(self, pic, scale):
        rs_pic = transforms.functional.resize(pic, scale)+(104, 117, 123)
        return transforms.functional.to_pil_image(rs_pic/255)

    def get_image_size(self):
        return self.image_size
