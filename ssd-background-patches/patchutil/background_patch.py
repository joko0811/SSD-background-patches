import torch
from torchvision import transforms

from .base_patch import BaseBackgroundManager


class BackgroundManager(BaseBackgroundManager):
    """
    適用するパッチ領域は画像の背景領域全てを用いる
    """

    def generate_patch(self, patch_size=(1237, 1649)):
        patch = torch.zeros((3,) + tuple(patch_size))
        return patch.clone()

    def transform_patch(self, patch, image_size, **kwargs):
        """
        Args:
            patch:
            image_size: (H,W)
        """
        patch = transforms.functional.resize(patch, image_size)
        mask = torch.ones((1,) + image_size).to(device=patch.device)
        return patch, mask
