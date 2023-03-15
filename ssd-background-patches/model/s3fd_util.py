import numpy as np

import torch
from torchvision import transforms

from model.S3FD import s3fd
from box.boxio import detections_base
from model.base_util import BaseUtilizer


class detections_s3fd(detections_base):
    def __init__(self, data, scale):
        xyxy = data[..., 1:]
        self.scale = scale

        super().__init__(data[..., 0], xyxy, is_xywh=False)

    def get_image_xyxy(self):
        return self.xyxy*self.scale


class detections_s3fd_ground_truth(detections_s3fd):
    def set_group_info(self, labels):
        self.group_labels = labels
        self.total_group = int(self.group_labels.max().item())+1


class detections_s3fd_loss(detections_s3fd):
    def set_loss_info(self, nearest_gt_idx):
        self.nearest_gt_idx = nearest_gt_idx


class S3fdResize:
    def __init__(self, scale_reference_size=(1700, 1200)):
        self.scale_reference = scale_reference_size[0]*scale_reference_size[1]

    def __call__(self, pic):
        width, height = transforms.functional.get_image_size(pic)
        x_area = width*height

        scale = np.sqrt(self.scale_reference/x_area)
        out_size = (round(height*scale), round(width*scale))
        return transforms.functional.resize(img=pic, size=out_size)


class S3fdUtilizer(BaseUtilizer):

    S3FD_TRANSFORMS = transforms.Compose([
        S3fdResize(),
        transforms.PILToTensor(),

    ])
    # caffe model RGB nomalization?
    S3FD_MASIC_NUMBER = torch.tensor([123., 117., 104.], dtype=torch.float)
    S3FD_NOMALIZED_ARRAY = S3FD_MASIC_NUMBER.detach().cpu().resolve_conj().resolve_neg().numpy()[
        :, np.newaxis, np.newaxis].astype('float32')

    def get_transform(self) -> transforms.Compose:
        return self.S3FD_TRANSFORMS

    def load_model(self, weight_path):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # Select device for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = s3fd.build_s3fd('test', 2)
        model.load_state_dict(torch.load(weight_path))

        return model.to(device)

    def make_detections_list(self, data_list, scale_list, detection_class: detections_s3fd, thresh):
        detections_list = list()
        for i, data in enumerate(data_list):
            extract_data = data[data[..., 0] >= thresh]
            if extract_data.nelement() != 0:
                detections_list.append(
                    detection_class(extract_data, scale_list[i]))
            else:
                detections_list.append(None)

        return detections_list

    def get_masic_number(self):
        return self.S3FD_NOMALIZED_ARRAY
