import glob
from PIL import Image

import torch
from torch.utils.data import Dataset

from box import boxio


class DirectoryImageWithDetectionDataset(Dataset):
    def __init__(self, image_path, detection_path, transform=None):
        self.image_path = image_path
        self.detection_path = detection_path
        self.files = sorted(glob.glob("%s/*.*" % self.image_path))
        self.detection_files = sorted(
            glob.glob("%s/*.*" % self.detection_path))
        self.transform = transform
        # TODO: Check if the names of image and mask match

    def __getitem__(self, index):
        image_path = self.files[index % len(self.files)]
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        # FIXME: tensor only
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        image_hw = image.shape[-2:]
        detection_path = self.detection_files[index % len(self.files)]
        label_list, xywh_list = boxio.parse_yolo(
            detection_path, image_hw)

        return image,  image_path, label_list, xywh_list

    def __len__(self):
        return len(self.files)
