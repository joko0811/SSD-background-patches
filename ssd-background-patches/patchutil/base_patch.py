import sys
import logging

import torch

from imageutil import imgseg
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

    def __init__(self):
        """
        Args:
          image_size: パッチを適用するデータセットの画像サイズのタプルまたはリスト(H,W)
          mode: test or train
        """
        # lower is better
        self.best_score = sys.maxsize

    def generate_patch(self):
        return

    def apply(self, patch, patch_mask, image_list, mask_list):
        """パッチ適用関数。複数毎に同時に適用できる"""
        return imgseg.composite_image_with_3_layer(
            image_list, mask_list, patch, patch_mask
        )

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
