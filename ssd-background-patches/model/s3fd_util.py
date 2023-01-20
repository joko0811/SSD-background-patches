import numpy as np

import torch

import cv2
from torchvision import transforms

from model.S3FD import s3fd
from box.boxio import detections_base

S3FD_TRANSFORMS = transforms.Compose([
    transforms.Resize((416, 416)),
])


def load_model(weights_path):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Select device for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = s3fd.build_s3fd('test', 2)
    model.load_state_dict(torch.load(weights_path))

    return model.to(device)


def image_setup(img):
    """
    Args:
        img: pil image
    Return:
        x: tensor
    """
    # I don't understand pre-processing.
    # TODO:Find minimum requirements
    # TODO:Eliminate numpy implementations

    if img.mode == 'L':
        img = img.convert('RGB')

    img = np.array(img)
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(
        1700 * 1200 / (img.shape[0] * img.shape[1]))
    image = cv2.resize(img, None, None, fx=max_im_shrink,
                       fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
    #image = cv2.resize(img, (640, 640))

    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    # RBG to BGR
    x = image[[2, 1, 0], :, :]

    x = x.astype('float32')
    x -= np.array([104., 117., 123.])[:, np.newaxis,
                                      np.newaxis].astype('float32')
    x = x[[2, 1, 0], :, :]

    x = torch.from_numpy(x).unsqueeze(0)

    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    return x, scale


"""
    def image_setup(image):
        max_im_shrink = math.sqrt(1700*1200/(image.shape[0]*image.shape[1]))
        transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (int(image.shape[0]*max_im_shrink), int(image.shape[1]*max_im_shrink)), interpolation=transforms.InterpolationMode.BILINEAR
            )
        ])(image)

        # HWC to CHW and RBG to BGR
        input_image = transform_image.permute((2, 0, 1)).flip(2).contiguous()
        input_image.permute((2, 1, 0))

        return input_image.to(dtype=torch.float)

        # np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')
"""


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
