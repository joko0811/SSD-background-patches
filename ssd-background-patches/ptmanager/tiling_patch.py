from math import ceil

import torch
from torchvision import transforms

from .base_patch import BaseBackgroundManager


class TilingBackgroundManager(BaseBackgroundManager):
    """
    適用するパッチ領域は画像の背景領域全てを用いる
    ただし背景領域適用前にパッチの敷き詰め処理を行う
    """

    def generate_patch(self):
        torch.manual_seed(0)
        patch = torch.rand(tuple((3,) + self.patch_size)) * 255
        return patch

    def calc_tiling_number(self, image_size, patch_size):
        """
        タイルを並べる数をH,Wそれぞれ求める
        画像のサイズに依存することに注意
        """
        tiling_number = (
            ceil(image_size[0] / patch_size[0]),  # H
            ceil(image_size[1] / patch_size[1]),  # W
        )
        return tiling_number

    def transform_patch(self, patch, image_size, **kwargs):
        """
        Args:
            patch:
            image_size: (H,W)
        """
        tiling_number = self.calc_tiling_number(image_size, patch.shape[1:])
        tiling_patch = transforms.functional.crop(
            patch.tile(tiling_number), 0, 0, image_size[0], image_size[1]
        )
        mask = torch.ones((1,) + image_size).to(device=patch.device)

        transformed_patch, _ = super().transform_patch(
            tiling_patch, image_size, **kwargs
        )
        return transformed_patch, mask


class RandomPutTilingManager(TilingBackgroundManager):
    def __init__(self, patch_size=(100, 200), put_num=10):
        """
        Args:
            patch_size: tuple(H,W)タイル一枚のサイズを指定する
        """
        self.put_num = put_num
        torch.manual_seed(0)

        super().__init__(patch_size)

    def transform_patch(self, patch, image_size, **kwargs):
        """パッチをランダムに配置する

        パッチを隙間なく配置した時の格子上に区切られた区画を元に配置
        格子をランダムに指定された数だけ選択する
        NOTE: 選択可能な領域がput_num以下である時は選択可能な領域分だけパッチを配置する

        Args:
            kwargs:
                seed: 乱数を用いるので再現性が必要なときに指定する。学習時はepoch数*画像インデックスをseedとするのが良い？
        """
        seed = kwargs["seed"] if "seed" in kwargs else None

        tiling_patch = torch.zeros((3,) + image_size).to(
            device=patch.device, dtype=patch.dtype
        )
        tiling_mask = torch.zeros((1,) + image_size).to(
            device=patch.device, dtype=torch.int
        )

        # タイルが縦何枚、横何枚配置できるか
        tiling_number = self.calc_tiling_number(image_size, patch.shape[1:])

        # パッチが縦Y枚目、横X枚目であるかを示すインデックスをランダムにput_num個分取得
        put_patch_idx = self._random_select_patch_idx(tiling_number, seed)

        # パッチの縦Y枚目、横X枚目の表現をtiling_patchの座標xyxyに変換
        put_patch_xyxy = self._patch_idx_2_tiling_idx(
            put_patch_idx, tiling_patch.shape[1:]
        )

        # 画像上
        tiling_patch = self._range_assignment(tiling_patch, patch, put_patch_xyxy)
        tiling_mask = self._range_assignment(
            tiling_mask, torch.ones((1,) + patch.shape[1:]), put_patch_xyxy
        ).to(dtype=bool)

        return tiling_patch, tiling_mask

    def _random_select_patch_idx(self, tiling_number, seed=None):
        """タイリング後のパッチを指定するインデックスをランダムに選択
        Args:
            tiling_number: タイリング後のパッチの縦横枚数(H,W)
            patch_num: パッチを選択する枚数
        Return:
            パッチが縦Y枚目、横X枚目であるかを示すインデックスをランダムにput_num個分取得
        """

        if seed is not None:
            torch.manual_seed(seed)

        # 0から(w * h - 1)までの整数をランダムな順序で並べたテンソルを生成し、n件まで取得
        tiling_num = torch.randperm(tiling_number[0] * tiling_number[1])[: self.put_num]
        # NOTE: 選択可能な領域がnum以下である時は選択可能な領域分だけパッチを配置する
        # 行と列のインデックスに変換
        tiling_h = (tiling_num / tiling_number[1]).to(dtype=torch.int)
        tiling_w = (tiling_num % tiling_number[1]).to(dtype=torch.int)
        tiling_idx = torch.stack((tiling_h, tiling_w), dim=1)

        return tiling_idx

    def _patch_idx_2_tiling_idx(self, patch_idx, image_size):
        """パッチの縦Y枚目、横X枚目の表現をtiling後の座標表現xyxyに変換
        Args:
            patch_idx: k個のタイリング後のパッチに対してのインデックス。縦N枚目、横M枚目でk*M*N
            image_size: タイリング後の形状H*W
        """
        tensor_patch_size = torch.tensor(self.patch_size)

        # タイルの縦Y枚目、横M枚目をput_tile_posのx,y座標に変換
        tile_yx1 = tensor_patch_size * patch_idx
        # 座標を画像のインデックス範囲内に収める
        tile_yx1[:, 0] = torch.clamp(tile_yx1[:, 0], min=0, max=image_size[0])  # y
        tile_yx1[:, 1] = torch.clamp(tile_yx1[:, 1], min=0, max=image_size[1])  # x

        # NOTE:リストの範囲指定用なので実際の座標ではない
        # 実際の座標を求めたいときは(tile_yx1 + self.patch_size - 1)
        tile_yx2 = tile_yx1 + tensor_patch_size

        # 座標を画像のインデックス範囲内に収める
        tile_yx2[:, 0] = torch.clamp(tile_yx2[:, 0], min=0, max=image_size[0])  # y
        tile_yx2[:, 1] = torch.clamp(tile_yx2[:, 1], min=0, max=image_size[1])  # x

        tile_xyxy = torch.stack(
            (tile_yx1[:, 1], tile_yx1[:, 0], tile_yx2[:, 1], tile_yx2[:, 0]), dim=1
        )

        return tile_xyxy

    def _range_assignment(self, target, source, range_list):
        assignmented_target = target.clone()
        for at_range in range_list:
            # range = [x1,y1,x2,y2]
            # target[:,y1:y2,x1:x2] = source
            assignmented_source = source.clone()
            if (at_range[3] - at_range[1]) != source.shape[1]:
                assignmented_source = assignmented_source[
                    :, : at_range[3] - at_range[1], :
                ]

            if (at_range[2] - at_range[0]) != source.shape[2]:
                assignmented_source = assignmented_source[
                    :, :, : at_range[2] - at_range[0]
                ]

            assignmented_target[
                :, at_range[1] : at_range[3], at_range[0] : at_range[2]
            ] = assignmented_source
        return assignmented_target


class ScalableTilingManager(TilingBackgroundManager):
    """
    基本はTilingBackgroundManagerと同様
    ただし、画像毎に一つの検出に合わせてパッチサイズが変動する
    """

    def __init__(self, patch_size=(100, 200), patch_det_area_ratio=6.66):
        """
        Args:
            patch_size: tuple(H,W)タイル一枚のサイズを指定する
        """
        self.patch_det_size_ratio = patch_det_area_ratio
        super().__init__(patch_size)

    def transform_patch(self, patch, image_size, **kwargs):
        """
        Args:
            patch:
            image_size: (H,W)
            kwargs:
                det_size: [N,2] N個の検出の幅と高さ(H,W)
        """
        if "det_size" not in kwargs:
            raise ValueError('argument "det_size" is must required')
        det_height = kwargs["det_size"][:, :, 0] * image_size[0]
        det_width = kwargs["det_size"][:, :, 1] * image_size[1]

        det_area = det_width * det_height
        patch_size_magnification = torch.sqrt(
            (6.66 * det_area[0, 0]) / (self.patch_size[0] * self.patch_size[1])
        )
        rescale_patch_size = (
            (torch.tensor(self.patch_size) * patch_size_magnification)
            .to(dtype=torch.int)
            .tolist()
        )

        rescaled_patch = transforms.functional.resize(patch, rescale_patch_size)

        return super().transform_patch(rescaled_patch, image_size)
