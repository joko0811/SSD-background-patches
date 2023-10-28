import random
import torch


def shielding_patch(patch, patch_mask, shielding_color=255, shielding_rate=0.0):
    # パッチマスクはCHW形式。座標のみを参照したいためチャンネル次元は削除
    patch_area_index = torch.nonzero(patch_mask.squeeze(0))
    # patch_area_indexはyx形式のn*2のテンソル

    shielding_area_pixel_num = torch.tensor(
        patch_area_index.shape[0] * shielding_rate
    ).to(dtype=torch.int64)

    if shielding_area_pixel_num < 1:
        return patch

    # reference_image = (
    #     torch.rand(size=patch.shape, dtype=patch.dtype)
    #     * (shielding_color_range[1] - shielding_color_range[0])
    #     + shielding_color_range[0]
    # )
    reference_image = torch.ones_like(patch) * shielding_color

    shielding_area_index = patch_area_index[:shielding_area_pixel_num, :]
    shielding_area_mask = torch.zeros_like(patch)
    shielding_area_mask[:, shielding_area_index[:, 0], shielding_area_index[:, 1]] = 1

    shielding_patch = torch.where(shielding_area_mask == 1, reference_image, patch)
    return shielding_patch


def shielding_patch_with_rectangle(
    patch, patch_mask, shielding_color=255, shielding_rate=0.0
):
    """パッチ領域に収まるランダムな矩形を遮蔽する
    パッチ領域が画像全域であるときしか処理できない
    """
    shielding_area_pixel_num = torch.tensor(
        patch.shape[1] * patch.shape[2] * shielding_rate
    ).to(dtype=torch.int64)

    height = round(
        random.uniform(
            (shielding_area_pixel_num / patch.shape[2]).item(), patch.shape[1]
        )
    )
    width = round(shielding_area_pixel_num.item() / height)

    y = int(random.uniform(0, patch.shape[1] - height))
    x = int(random.uniform(0, patch.shape[2] - width))

    shielded_patch = patch.clone()
    shielded_patch[:, y : y + height, x : x + width] = shielding_color

    return shielded_patch


if __name__ == "__main__":
    from torchvision.transforms.functional import to_pil_image

    patch = torch.ones((3, 300, 300))
    patch *= 0.5
    patch_mask = torch.zeros((1, 300, 300))
    patch_mask[:, 20:60, 10:60] = 1
    patch_mask[:, 70:150, 80:150] = 1
    patch_mask[:, 230:280, 230:250] = 1

    # shielded_patch = shielding_patch(patch, patch_mask, shielding_rate=0.1)
    shielded_patch = shielding_patch_with_rectangle(
        patch, patch_mask, shielding_rate=0.5
    )

    to_pil_image(patch).save("patch.png")
    to_pil_image(patch_mask).save("patch_mask.png")
    to_pil_image(shielded_patch).save("shielded_patch.png")
