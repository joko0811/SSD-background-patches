import glob
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

from box import boxio


class TrainBackGroundDataset(Dataset):
    """敵対的背景画像の学習用データセット
    顔領域を含む画像、顔を対象とした時のマスク画像、顔領域の検出がセットになっている
    データセットに要求される構造は以下である。ファイル名をIDとして、三つのディレクトリは同一IDを共有する（ディレクトリ以下には拡張子を除いて同じ名前のファイルがあることを期待する）

    face_files: データセットの主体。顔を含む画像が配置される
    mask_files: 顔画像に対するマスク画像
    detection_files: 真の顔領域が存在する
    """

    def __init__(
        self, face_path, mask_path, detection_path, max_iter=None, transform=None
    ):
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
            self.face_files = self.face_files[: self.max_iter]
            self.mask_files = self.mask_files[: self.max_iter]
            self.detection_files = self.detection_files[: self.max_iter]

        self.transform = transform

    def _check_dataset(self):
        """顔画像、マスク画像、検出の組がつくれるか検証"""
        face_checker = [
            os.path.splitext(os.path.basename(ff))[0] for ff in self.face_files
        ]
        mask_checker = [
            os.path.splitext(os.path.basename(mf))[0] for mf in self.mask_files
        ]
        detection_checker = [
            os.path.splitext(os.path.basename(df))[0] for df in self.detection_files
        ]

        if not (len(face_checker) == len(mask_checker) == len(detection_checker)):
            raise Exception

        for i in range(len(face_checker)):
            if not (face_checker[i] == mask_checker[i] == detection_checker[i]):
                raise Exception(
                    f"Filename does not match: {face_checker[i]}, {mask_checker[i]}, {detection_checker[i]}"
                )

    def __getitem__(self, index):

        face_path = self.face_files[index % len(self.face_files)]
        mask_path = self.mask_files[index % len(self.mask_files)]
        detection_path = self.detection_files[index % len(
            self.detection_files)]

        face_image = Image.open(face_path)
        mask_image = Image.open(mask_path).convert("1")
        det = boxio.parse_detections(detection_path)
        if det is not None:
            conf, xyxy = det
        else:
            conf = torch.tensor([])
            xyxy = torch.tensor([])

        width, height = face_image.size

        if self.transform is not None:
            face_image = self.transform(face_image)
            mask_image = self.transform(mask_image)

        face_image_info = {}
        face_image_info["width"] = torch.tensor([width])
        face_image_info["height"] = torch.tensor([height])
        face_image_info["conf"] = conf
        face_image_info["xyxy"] = xyxy
        face_image_info["path"] = face_path

        return ((face_image, mask_image), face_image_info)

    def __len__(self):
        return len(self.face_files)


# TODO: マスク画像生成、検出作成のコードも記載する


class TestBackGroundDataset(Dataset):
    """敵対的背景画像の学習用データセット
    顔領域を含む画像、顔を対象とした時のマスク画像、顔領域の検出がセットになっている
    """

    def __init__(self, face_path, mask_path, max_iter=None, transform=None):
        self.face_path = face_path
        self.mask_path = mask_path

        self.face_files = sorted(glob.glob("%s/*.*" % self.face_path))
        self.mask_files = sorted(glob.glob("%s/*.*" % self.mask_path))

        try:
            self._check_dataset()
        except Exception as e:
            print(e)

        if max_iter is not None:
            self.max_iter = min(max_iter, len(self.face_files))
            self.face_files = self.face_files[: self.max_iter]
            self.mask_files = self.mask_files[: self.max_iter]

        self.transform = transform

    def _check_dataset(self):
        """顔画像、マスク画像、検出の組がつくれるか検証"""
        face_checker = [
            os.path.splitext(os.path.basename(ff))[0] for ff in self.face_files
        ]
        mask_checker = [
            os.path.splitext(os.path.basename(mf))[0] for mf in self.mask_files
        ]

        if not (len(face_checker) == len(mask_checker)):
            raise Exception

        for i in range(len(face_checker)):
            if not (face_checker[i] == mask_checker[i]):
                raise Exception(
                    f"Filename does not match: {face_checker[i]}, {mask_checker[i]}"
                )

    def __getitem__(self, index):

        face_path = self.face_files[index % len(self.face_files)]
        mask_path = self.mask_files[index % len(self.mask_files)]

        face_image = Image.open(face_path)
        mask_image = Image.open(mask_path).convert("1")

        width, height = face_image.size

        if self.transform is not None:
            face_image = self.transform(face_image)
            mask_image = self.transform(mask_image)

        face_image_info = {}
        face_image_info["width"] = torch.tensor([width])
        face_image_info["height"] = torch.tensor([height])

        return ((face_image, mask_image), face_image_info)

    def __len__(self):
        return len(self.face_files)


# TODO: マスク画像生成、検出作成のコードも記載する
