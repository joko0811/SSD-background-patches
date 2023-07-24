import torch
from torchvision import transforms

from .base_patch import BaseBackgroundManager


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
        patch = transforms.functional.resize(patch, image_size)
        mask = torch.ones((1,) + image_size).to(device=patch.device)
        return patch, mask
