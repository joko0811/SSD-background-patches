import glob
import os
from PIL import Image
from torch.utils.data import Dataset

from box import boxio


class BackGroundTrainDataset(Dataset):
    """敵対的背景画像の学習用データセット
    顔領域を含む画像、顔を対象とした時のマスク画像、顔領域の検出がセットになっている
    """

    def __init__(self, face_path, mask_path, detect_path, max_iter=None, transform=None):
        self.face_path = face_path
        self.mask_path = mask_path
        self.detect_path = detect_path

        self.face_files = sorted(glob.glob("%s/*.*" % self.face_path))
        self.mask_files = sorted(glob.glob("%s/*.*" % self.mask_path))
        self.detect_files = sorted(glob.glob("%s/*.*" % self.detect_path))

        try:
            self._check_dataset()
        except Exception as e:
            print(e)

        if max_iter is not None:
            self.max_iter = max_iter
            self.face_files = self.face_files[:self.max_iter]
            self.mask_files = self.mask_files[:self.max_iter]
            self.detect_files = self.detect_files[:self.max_iter]

        self.transform = transform

    def _check_dataset(self):
        """顔画像、マスク画像、検出の組がつくれるか検証"""
        face_checker = [os.path.basename(ff) for ff in self.face_files]
        mask_checker = [os.path.basename(mf) for mf in self.mask_files]
        detect_checker = [os.path.basename(df) for df in self.detect_files]

        if not (len(face_checker) == len(mask_checker) == len(detect_checker)):
            raise Exception

        for i in range(len(face_checker)):
            if not (face_checker[i] == mask_checker[i] == detect_checker[i]):
                raise Exception(
                    f'Filename does not match: {face_checker[i]}, {mask_checker[i]}, {detect_checker[i]}')

    def __getitem__(self, index):

        face_path = self.face_files[index % len(self.face_files)]
        mask_path = self.mask_files[index % len(self.mask_files)]
        detect_path = self.detect_files[index % len(self.detect_files)]

        face_image = Image.open(face_path)
        mask_image = Image.open(mask_path)
        conf, xyxy_list = boxio.parse_detections(detect_path)

        if self.transform is not None:
            face_image = self.transform(face_image)
            mask_image = self.transform(mask_image)

        return ((face_image, mask_image), (conf, xyxy_list))

    def __len__(self):
        return len(self.files)

# TODO: マスク画像生成、検出作成のコードも記載する
