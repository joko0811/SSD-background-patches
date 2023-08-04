import torch

from .base_patch import BaseBackgroundManager


class PositionalBackgroundManager(BaseBackgroundManager):
    def generate_patch(self, patch_size=(40, 40)):
        patch = torch.zeros((3,) + tuple(patch_size))
        return patch.clone()

    def transform_patch(self, patch, image_size, patch_coorinate=(0, 0)):
        """
        Args:
            patch_coorinate: (y,x)
        """
        mask = torch.zeros((1,) + image_size).to(device=patch.device)

        compose_patch = torch.zeros((3,) + image_size).to(
            device=patch.device, dtype=torch.float
        )
        compose_patch[
            :,
            patch_coorinate[0] : patch_coorinate[0] + patch.shape[1],
            patch_coorinate[1] : patch_coorinate[1] + patch.shape[2],
        ] = patch
        mask[
            :,
            patch_coorinate[0] : patch_coorinate[0] + patch.shape[1],
            patch_coorinate[1] : patch_coorinate[1] + patch.shape[2],
        ] = 1

        return (compose_patch, mask.clone())
