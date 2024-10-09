import torch
from torchvision import transforms

from model.Retinaface import detect
from model.Retinaface.data import cfg_re50
from model.Retinaface.models.retinaface import RetinaFace
from model.Retinaface.layers.functions.prior_box import PriorBox
from model.Retinaface.utils.box_utils import decode
from model.Retinaface.utils.box_utils import nms

from model.base_util import BackgroundBaseTrainer
from detection.detection_base import DetectionsBase
from omegaconf import DictConfig


class RetinaResize:
    def __init__(self, scale_reference_size=(840, 840)):
        self.scale_reference_size = scale_reference_size

    def __call__(self, pic):
        width, height = transforms.functional.get_image_size(pic)
        ref_area = self.scale_reference_size[0] * self.scale_reference_size[1]

        if pic.shape[0] == 3:
            conv_pic = pic - torch.tensor([104, 117, 123]).view(3, 1, 1)
        else:
            conv_pic = pic.clone()

        if width * height < ref_area:
            return transforms.functional.resize(
                img=conv_pic, size=self.scale_reference_size
            )
        else:
            return conv_pic


class RetinaTrainer(BackgroundBaseTrainer):
    RETINA_SCALE_REFERENCE = (840, 840)
    RETINA_TRANSFORMS = transforms.Compose(
        [
            transforms.PILToTensor(),
            RetinaResize(scale_reference_size=RETINA_SCALE_REFERENCE),
        ]
    )

    def __init__(self, model_conf: DictConfig, dataset_factory):
        self.model_conf = model_conf
        dataset = dataset_factory(transform=self.RETINA_TRANSFORMS)
        self.dataloader = torch.utils.data.DataLoader(dataset)

    def get_dataloader(self):
        return self.dataloader

    def load_model(self, device=None, mode="test"):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net = RetinaFace(cfg=cfg_re50, phase=mode)
        model = detect.load_model(net, self.model_conf.weight_path, False)

        return model.to(device)

    def make_detections_list(self, data_list, thresh=0.6, scale=None, img_size=None):
        loc_list, conf_list, _ = data_list
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        detections_list = list()

        for i in range(len(loc_list)):
            priorbox = PriorBox(cfg_re50, image_size=img_size[i])
            priors = priorbox.forward().to(device)
            boxes = decode(loc_list[i], priors, cfg_re50["variance"])
            conf = conf_list[i, :, 1]

            ext1_idx, _ = nms(boxes, conf, overlap=0.4, top_k=750)
            ext1_conf = conf[ext1_idx].clone()
            ext1_boxes = boxes[ext1_idx].clone()

            ext2_idx = torch.where(ext1_conf > thresh)
            ext2_conf = ext1_conf[ext2_idx].clone()
            ext2_boxes = ext1_boxes[ext2_idx].clone()

            if ext2_conf.nelement() != 0:
                detections_list.append(
                    DetectionsBase(ext2_conf, ext2_boxes, is_xywh=False)
                )
            else:
                detections_list.append(None)

        return detections_list

    def transformed2pil(self, pic, scale=None):
        conv_pic = pic + torch.tensor([104, 117, 123]).view(3, 1, 1).to(
            dtype=pic.dtype, device=pic.device
        )
        if scale is None:
            return transforms.functional.to_pil_image(conv_pic / 255)
        rs_pic = transforms.functional.resize(conv_pic, scale)
        return transforms.functional.to_pil_image(rs_pic / 255)
