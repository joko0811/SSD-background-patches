import glob
import os
from PIL import Image
from torch.utils.data import Dataset

from box import boxio


class TrainBackGroundDataset(Dataset):
    """敵対的背景画像の学習用データセット
    顔領域を含む画像、顔を対象とした時のマスク画像、顔領域の検出がセットになっている
    """

    def __init__(self, face_path, mask_path, detection_path, max_iter=None, transform=None):
        self.face_path = face_path
        self.mask_path = mask_path
        self.detection_path = detection_path

        self.face_files = sorted(glob.glob("%s/*.*" % self.face_path))
        self.mask_files = sorted(glob.glob("%s/*.*" % self.mask_path))
        self.detection_files = sorted(
            glob.glob("%s/*.*" % self.detection_path))

        try:
            self._check_dataset()
        except Exception as e:
            print(e)

        if max_iter is not None:
            self.max_iter = min(max_iter, len(self.face_files))
            self.face_files = self.face_files[:self.max_iter]
            self.mask_files = self.mask_files[:self.max_iter]
            self.detection_files = self.detection_files[:self.max_iter]

        self.transform = transform

    def _check_dataset(self):
        """顔画像、マスク画像、検出の組がつくれるか検証"""
        face_checker = [os.path.basename(ff) for ff in self.face_files]
        mask_checker = [os.path.basename(mf) for mf in self.mask_files]
        detection_checker = [os.path.basename(
            df) for df in self.detection_files]

        if not (len(face_checker) == len(mask_checker) == len(detection_checker)):
            raise Exception

        for i in range(len(face_checker)):
            if not (face_checker[i] == mask_checker[i] == detection_checker[i]):
                raise Exception(
                    f'Filename does not match: {face_checker[i]}, {mask_checker[i]}, {detection_checker[i]}')

    def __getitem__(self, index):

        face_path = self.face_files[index % len(self.face_files)]
        mask_path = self.mask_files[index % len(self.mask_files)]
        detection_path = self.detection_files[index % len(
            self.detection_files)]

        face_image = Image.open(face_path)
        mask_image = Image.open(mask_path)
        conf, xyxy = boxio.parse_detections(detection_path)

        width, height = face_image.size

        if self.transform is not None:
            face_image = self.transform(face_image)
            mask_image = self.transform(mask_image)

        face_image_info = {}
        face_image_info['width'] = width
        face_image_info['height'] = height
        face_image_info['conf'] = conf
        face_image_info['xyxy'] = xyxy

        return ((face_image, mask_image), face_image_info)

    def __len__(self):
        return len(self.face_files)

# TODO: マスク画像生成、検出作成のコードも記載する
