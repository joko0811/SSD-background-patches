import sys
import logging
import random

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

    def __init__(self, patch_size=(100, 200)):
        """
        Args:
            patch_size: tuple(H,W)タイル一枚のサイズを指定する
            mode: test or train
        """
        # lower is better
        self.best_score = sys.maxsize
        self.patch_size = patch_size
        self.patch_postprosesser = None

    def generate_patch(self):
        return

    def set_patch_postprosesser(self, patch_postprosesser):
        self.patch_postprosesser = patch_postprosesser

    def apply(self, patch, patch_mask, image_list, mask_list):
        """パッチ適用関数。複数毎に同時に適用できる"""
        if self.patch_postprosesser is not None:
            middle_image = self.patch_postprosesser(patch, patch_mask)
        else:
            middle_image = patch

        return imgseg.composite_image_with_3_layer(
            image_list, mask_list, middle_image, patch_mask
        )

    def transform_patch(self, patch, image_size, **kwargs):
        """パッチを適用するために必要な変形を行う
        同時にパッチを適用する領域を指定するマスク画像も生成する
        Args:
            patch:
            image_size: (H,W)
        """
        mask = torch.ones((1,) + tuple(image_size)).to(device=patch.device)
        return patch, mask.clone()

    def generate_kwargs_of_transform_patch(
        self, image_size, patch_size, det_size, seed=None
    ):
        """transform_patchに渡すkwargsを生成する
        Args:
            image_size: (H,W)
            patch_size: (H,W)
            det_size: [N,2] N個の検出について、高さ、幅の組(H,W)をそれぞれ保持するテンソル
            seed: シードを利用しない場合はNoneを指定する
        """

        args_of_tpatch = {}

        # パッチのランダムな座標変動
        # RandomPutTilingManagerなどのパッチをランダム配置する場合に使用する
        args_of_tpatch["patch_coordinate"] = (
            int(random.random() * (image_size[0] - patch_size[0])),
            int(random.random() * (image_size[1] - patch_size[1])),
        )
        # 検出の高さと幅
        # ScalableTilingManagerなどのパッチサイズが検出サイズに依存する場合に使用する
        args_of_tpatch["det_size"] = det_size

        # ランダム生成する際のシード値
        # RandomPutTilingManagerなどの変形時にランダム性が必要である場合に使用する
        args_of_tpatch["seed"] = seed

        return args_of_tpatch

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
