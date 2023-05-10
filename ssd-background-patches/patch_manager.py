import torch

from imageutil import imgseg
from torchvision import transforms


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
    """マスク画像をもとに背景画像合成を行う
    """

    def __init__(self, image_size):
        """
        Args:
          image_size: パッチを適用するデータセットの画像サイズのタプルまたはリスト(H,W)
        """
        self.image_size = image_size

    def apply(self, patch, image_list, mask_list):
        """パッチ適用関数。複数毎に同時に適用できる"""
        return imgseg.composite_image(image_list, patch, mask_list)


class BackgroundManager(BaseBackgroundManager):
    """
    適用するパッチ領域は画像の背景領域全てを用いる
    """

    def generate_patch(self):
        patch = torch.zeros((3,)+tuple(self.image_size))
        return patch


class TilingBackgroundManager(BaseBackgroundManager):
    """
    適用するパッチ領域は画像の背景領域全てを用いる
    ただし背景領域適用前にパッチの敷き詰め処理を行う
    """

    TILE_SIZE = (100, 200)  # (H,W)

    def __init__(self, image_size):
        self.tiling_number = (
            int(image_size[0]/self.TILE_SIZE[0]), int(image_size[1]/self.TILE_SIZE[1]))
        super().__init__(image_size)

    def generate_patch(self):
        patch = torch.zeros((3,) + self.TILE_SIZE)
        return patch

    def apply(self, patch, image_list, mask_list):
        tiling_patch = patch.tile(self.tiling_number)
        resized_image_list = transforms.functional.resize(
            image_list, tiling_patch.shape[1:])
        resized_mask_list = transforms.functional.resize(
            mask_list, tiling_patch.shape[1:])
        return super().apply(resized_image_list, tiling_patch, resized_mask_list)
