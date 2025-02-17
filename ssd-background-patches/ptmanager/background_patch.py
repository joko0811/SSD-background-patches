import torch
from torchvision import transforms

from .base_patch import BaseBackgroundManager


class BackgroundManager(BaseBackgroundManager):
    """
    適用するパッチ領域は画像の背景領域全てを用いる
    """

    def generate_patch(self):
        patch = torch.zeros((3,) + tuple(self.patch_size))
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


class AllBackgroundManager(BackgroundManager):

    def apply(self, patch, patch_mask, image_list, mask_list):
        applied_patch = patch.repeat((image_list.shape[0], 1, 1, 1))
        return applied_patch
