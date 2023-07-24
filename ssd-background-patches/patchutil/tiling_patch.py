from math import ceil

import torch
from torchvision import transforms

from .base_patch import BaseBackgroundManager


class TilingBackgroundManager(BaseBackgroundManager):
    """
    適用するパッチ領域は画像の背景領域全てを用いる
    ただし背景領域適用前にパッチの敷き詰め処理を行う
    """

    def __init__(self, tile_size=(100, 200)):
        """
        Args:
            tile_size: tuple(H,W)タイル一枚のサイズを指定する
        """
        self.tile_size = tile_size
        super().__init__()

    def generate_patch(self):
        patch = torch.zeros(tuple((3,) + self.tile_size))
        return patch

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
