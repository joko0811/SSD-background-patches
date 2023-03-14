import numpy as np

import torch

import cv2
from PIL import Image
from torchvision import transforms

from model.S3FD import s3fd
from box.boxio import detections_base

from typing import Tuple

S3FD_TRANSFORMS = transforms.Compose([
    transforms.Resize((1700, 1200)),
    transforms.PILToTensor(),
])

# caffe model RGB nomalization?
S3FD_MASIC_NUMBER = torch.tensor([123., 117., 104.], dtype=torch.float)
S3FD_NOMALIZED_ARRAY = S3FD_MASIC_NUMBER.detach().cpu().resolve_conj().resolve_neg().numpy()[
    :, np.newaxis, np.newaxis].astype('float32')
S3FD_IMAGE_MIN = torch.tensor([0., 0., 0.])-S3FD_MASIC_NUMBER
S3FD_IMAGE_MAX = torch.tensor([255., 255., 255.]) - S3FD_MASIC_NUMBER


def load_model(weight_path):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Select device for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = s3fd.build_s3fd('test', 2)
    model.load_state_dict(torch.load(weight_path))

    return model.to(device)


def image_list_encode(pil_image_list: Image.Image, is_mask=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """image_encode wrap supporting list encoding.
    Args:
        pil_image_list: list of pil image
    Return:
        tensor_image: Tensor list of s3fd input representation
    """
    setuped_image_list = list()
    scale_list = list()
    for pil_image in pil_image_list:
        image, scale = image_encode(pil_image, is_mask)
        setuped_image_list.append(image)
        scale_list.append(scale)

    return torch.cat(setuped_image_list).contiguous(), torch.cat(scale_list).contiguous()


def image_encode(pil_image, is_mask=False):
    """I don't understand pre-processing.
    Args:
        pil_image: a pil image
    Return:
        tensor_image: Tensor of s3fd input representation
    """

    # pil to ndarray
    if pil_image.mode == 'L':
        # 8-bit pixels, black and white
        pil_image = pil_image.convert('RGB')
    np_image = np.array(pil_image)

    # resize
    height, width, _ = np_image.shape
    max_im_shrink = np.sqrt(
        1700 * 1200 / (height * width))
    resize_image = cv2.resize(np_image, None, None, fx=max_im_shrink,
                              fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)

    # HWC to CHW
    chw_image = hwc2chw(resize_image).astype('float32')

    if not is_mask:
        # nomalization
        nomalized_image = chw_image-S3FD_NOMALIZED_ARRAY
    else:
        nomalized_image = chw_image

    # numpy to tensor
    tensor_image = torch.from_numpy(nomalized_image).unsqueeze(0)

    # old width height array
    scale = torch.Tensor([width, height,
                          width, height]).unsqueeze(0)

    return tensor_image, scale


def image_decode(tensor_image: torch.Tensor, scale=None):
    """
    Args:
        tensor_image: Tensor of s3fd input representation
        scale: image_encode() output
    Return:
        image: pil image
    """
    if scale is None:
        scale = torch.tensor([1700, 1200])

    np_image = tensor_image.detach().cpu().resolve_conj().resolve_neg().numpy()
    before_nomalized_image = np_image+S3FD_NOMALIZED_ARRAY

    # CHW to HWC
    hwc_image = chw2hwc(before_nomalized_image)

    max_im_shrink = float(
        1/np.sqrt(1700*1200/(scale[0].item() * scale[1].item())))
    resize_image = cv2.resize(hwc_image, None, None, fx=max_im_shrink,
                              fy=max_im_shrink, interpolation=cv2.INTER_LINEAR).astype('uint8')
    pil_image = Image.fromarray(resize_image)

    return pil_image


def hwc2chw(np_image):
    return np.swapaxes(np.swapaxes(np_image, 1, 2), 0, 1)


def chw2hwc(np_image):
    return np.swapaxes(np.swapaxes(np_image, 0, 1), 1, 2)


def make_detections_list(data_list, scale_list, detection_class, thresh):
    detections_list = list()
    for i, data in enumerate(data_list):
        extract_data = data[data[..., 0] >= thresh]
        if extract_data.nelement() != 0:
            detections_list.append(
                detection_class(extract_data, scale_list[i]))
        else:
            detections_list.append(None)

    return detections_list


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
