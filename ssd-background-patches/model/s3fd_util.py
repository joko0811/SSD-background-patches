import numpy as np

import torch

import cv2
from PIL import Image
from torchvision import transforms

from model.S3FD import s3fd
from box.boxio import detections_base

S3FD_TRANSFORMS = transforms.Compose([
    transforms.Resize((416, 416)),
])

# caffe model nomalization?
S3FD_NOMALIZATION_NDARRAY = np.array([123., 117., 104.])[
    :, np.newaxis, np.newaxis].astype('float32')


def load_model(weight_path):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Select device for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = s3fd.build_s3fd('test', 2)
    model.load_state_dict(torch.load(weight_path))

    return model.to(device)


def image_list_encode(pil_image_list):
    """image_encode wrap supporting list encoding.
    Args:
        pil_image_list: list of pil image
    Return:
        tensor_image: Tensor list of s3fd input representation
    """
    setuped_image_list = list()
    scale_list = list()
    for pil_image in pil_image_list:
        image, scale = image_encode(pil_image)
        setuped_image_list.append(image.squeeze(0))
        scale_list.append(scale)

    return torch.cat(setuped_image_list).contiguous(), torch.cat(scale).contiguous()


def image_encode(pil_image):
    """I don't understand pre-processing.
    Args:
        pil_image: a pil image
    Return:
        tensor_image: Tensor of s3fd input representation
    """

    # pil to ndarray
    if pil_image.mode == 'L':
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

    # nomalization
    nomalized_image = chw_image-S3FD_NOMALIZATION_NDARRAY

    # numpy to tensor
    tensor_image = torch.from_numpy(nomalized_image).unsqueeze(0)

    # old width height array
    scale = torch.Tensor([width, height,
                          width, height])

    return tensor_image, scale


def image_decode(tensor_image: torch.Tensor, scale):
    """
    Args:
        tensor_image: Tensor of s3fd input representation
        scale: image_encode() output
    Return:
        image: pil image
    """

    np_image = tensor_image.detach().cpu().resolve_conj().resolve_neg().numpy()
    before_nomalized_image = np_image+S3FD_NOMALIZATION_NDARRAY

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


class detections_s3fd(detections_base):
    # yolo_out=[x,y,w,h,confidence_score,class_scores...]
    # nms_out=[x1,y1,x2,y2,x,y,w,h,confidence_score,class_scores...]
    def __init__(self, data, scale):
        self.conf = data[..., 0]
        self.xyxy = data[..., 1:]*scale

        self.total_det = self.xyxy.shape[0]

        super().__init__(self.conf, self.xyxy, is_xywh=False)


class detections_s3fd_ground_truth(detections_s3fd):
    def set_group_info(self, labels):
        self.group_labels = labels
        self.total_group = int(self.group_labels.max().item())+1


class detections_yolo_loss(detections_s3fd):
    def set_loss_info(self, nearest_gt_idx):
        self.nearest_gt_idx = nearest_gt_idx
