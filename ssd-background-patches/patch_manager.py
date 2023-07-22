import sys
import logging
import os
from math import ceil

import torch

from imageutil import imgseg
from torchvision import transforms

# ベストパッチの評価用
from evaluation.detection import f1, precision, recall


class BasePatchManager:
    """
    初期化時のパッチのサイズの設定と、データセットの画像毎に行うパッチ適用を行う
    パッチに対して逆伝播を行うためクラス内部ではパッチを保持しないようにする
    """

    def __init__(self):
        pass

    def generate_patch(self):
        """
        パッチサイズなどの初期化を行う。返り値としてパッチを渡す
        """
        pass

    def apply(self, patch):
        """パッチ適用関数。複数毎に同時に適用できる"""
        # self.patch_transform(patch)
        pass


class BaseBackgroundManager:
    """マスク画像をもとに背景画像合成を行う"""

    def __init__(self, mode):
        """
        Args:
          image_size: パッチを適用するデータセットの画像サイズのタプルまたはリスト(H,W)
          mode: test or train
        """
        self.mode = mode
        # lower is better
        self.best_score = sys.maxsize

    def generate_patch(self):
        return

    def apply(self, patch, image_list, mask_list):
        """パッチ適用関数。複数毎に同時に適用できる"""
        return imgseg.composite_image(image_list, patch, mask_list)

    def transform_patch(self, patch, image_size):
        """
        Args:
            patch:
            image_size: (H,W)
        """
        return patch

    def save_best_image(self, patch, path, ground_trhuth, tp, fp, fn):
        logging.info(
            "tp: " + str(tp) + ", fp: " + str(fp) + ", gt: " + str(len(ground_trhuth))
        )
        precision_score = precision(tp, fp)
        recall_score = recall(tp, fn)

        f1_score = f1(precision_score, recall_score)
        # duq = data_utility_quority(len(ground_trhuth), tp, fp)
        torch.save(patch, path)
        out_str = "f1: " + str(f1_score)
        if f1_score <= self.best_score:
            self.best_score = f1_score
            out_str += " update best score"
        logging.info(out_str)


class BackgroundManager(BaseBackgroundManager):
    """
    適用するパッチ領域は画像の背景領域全てを用いる
    """

    def generate_patch(self):
        patch = torch.zeros((3,) + tuple(self.image_size))
        return patch.clone()

    def transform_patch(self, patch, image_size):
        """
        Args:
            patch:
            image_size: (H,W)
        """
        return transforms.functional.resize(patch, image_size)


class TilingBackgroundManager(BaseBackgroundManager):
    """
    適用するパッチ領域は画像の背景領域全てを用いる
    ただし背景領域適用前にパッチの敷き詰め処理を行う
    """

    def __init__(self, mode, tile_size=(100, 200)):
        """
        Args:
            tile_size: tuple(H,W)タイル一枚のサイズを指定する
        """

        self.tile_size = tile_size
        super().__init__(mode)

    def generate_patch(self):
        patch = torch.zeros((3,) + self.tile_size)
        return patch

    def apply(self, patch, image_list, mask_list):
        if self.mode == "train":
            tiling_patch = self.transform_patch(patch)
        else:
            tiling_patch = patch.clone()
        resized_image_list = transforms.functional.resize(
            image_list, tiling_patch.shape[1:]
        )
        resized_mask_list = transforms.functional.resize(
            mask_list, tiling_patch.shape[1:]
        )
        return super().apply(tiling_patch, resized_image_list, resized_mask_list)

    def transform_patch(self, patch, image_size):
        """
        Args:
            patch:
            image_size: (H,W)
        """
        tiling_number = (
            ceil(image_size[0] / self.tile_size[0]),
            ceil(image_size[1] / self.tile_size[1]),
        )
        tiling_patch = transforms.functional.crop(
            patch.tile(tiling_number), 0, 0, image_size[0], image_size[1]
        )
        return tiling_patch
