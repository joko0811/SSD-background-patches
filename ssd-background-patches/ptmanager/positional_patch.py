import torch

from .base_patch import BaseBackgroundManager


class PositionalBackgroundManager(BaseBackgroundManager):
    # def generate_patch(self, patch_size=(40, 40)):
    def generate_patch(self):
        patch = torch.zeros((3,) + tuple(self.patch_size))
        return patch.clone()

    def transform_patch(self, patch, image_size, **kwargs):
        """
        Args:
            kwargs:
            patch_coorinate: (y,x)パッチを配置する座標を指定する。デフォルトは(0,0)
        """
        patch_coordinate = (
            kwargs["patch_coordinate"] if "patch_coordinate" in kwargs else (0, 0)
        )

        mask = torch.zeros((1,) + image_size).to(device=patch.device)

        compose_patch = torch.zeros((3,) + image_size).to(
            device=patch.device, dtype=torch.float
        )
        compose_patch[
            :,
            patch_coordinate[0] : patch_coordinate[0] + patch.shape[1],
            patch_coordinate[1] : patch_coordinate[1] + patch.shape[2],
        ] = patch
        mask[
            :,
            patch_coordinate[0] : patch_coordinate[0] + patch.shape[1],
            patch_coordinate[1] : patch_coordinate[1] + patch.shape[2],
        ] = 1

        transformed_patch, _ = self.super().transform_patch(
            compose_patch, image_size, **kwargs
        )

        return (transformed_patch, mask.clone())
