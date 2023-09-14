# This file is a reproduction implementation of
# Exploring the Vulnerability of Single Shot Module in Object Detectors via Imperceptible Background Patches
import os

from PIL import Image
import numpy as np
import hydra
from omegaconf import DictConfig

import torch
import torch.optim as optim
from torchvision import transforms

from tensorboardX import SummaryWriter
from tqdm import tqdm

import proposed_func as pf
from model import s3fd_util
from loss.li2019_improved import total_loss
from util import clustering
from imageutil import imgdraw, imgseg


def train_adversarial_image(image, mask_image, config: DictConfig, tbx_writer=None):
    max_epoch = config.max_epoch  # default 300

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weightfile_path = os.path.join(os.getcwd(), config.model.weight_path)
    model = s3fd_util.load_model(weightfile_path)
    model.eval()

    # 敵対的背景
    # adv_background_image = torch.zeros((3, 416, 416), device=device)
    # (1237,1649) is size of dataset image in S3FD representation
    s3fd_adv_background_image = torch.zeros((1, 3, 1237, 1649), device=device)
    bg_scale = torch.tensor([1649, 1237, 1649, 1237]).to(device=device)

    optimizer = optim.Adam([s3fd_adv_background_image])

    # 画像の準備とground truthの作成
    with torch.no_grad():
        encoded_tuple = s3fd_util.image_encode(image)
        s3fd_image = encoded_tuple[0].to(device=device, dtype=torch.float)
        scale = encoded_tuple[1].to(device=device, dtype=torch.float)

        s3fd_mask_image = s3fd_util.image_encode(mask_image, is_mask=True)[0].to(
            device=device, dtype=torch.float
        )

        gt_output = model(s3fd_image)
        gt_detections = s3fd_util.make_detections_list(gt_output, config.model_thresh)[
            0
        ]

        if gt_detections is None:
            # 元画像の検出がない場合は敵対的画像を生成できない
            return s3fd_adv_background_image

        # ground truthesをクラスタリング
        group_labels = clustering.object_grouping(
            gt_detections.xywh.detach().cpu().numpy()
        )
        gt_detections.set_group_info(
            torch.from_numpy(group_labels.astype(np.float32)).to(device)
        )

    # グループ毎に割り当てられるパッチ枚数
    n_b = config.n_b  # default 3
    background_patch_boxes = torch.zeros(
        (gt_detections.total_group * n_b, 4), device=device
    )

    optimizer = optim.Adam([s3fd_adv_background_image])

    for perturbate_iter in tqdm(range(max_epoch), leave=(tbx_writer is not None)):
        s3fd_adv_background_image.requires_grad = True

        s3fd_adv_image = imgseg.composite_image(
            s3fd_image, s3fd_adv_background_image[0], s3fd_mask_image
        )

        # perturbate_iter回目のパッチ適用画像から物体検出する
        output = model(s3fd_adv_image)
        detections = s3fd_util.make_detections_list(output, 0.6)[0]

        if detections is None:
            # TODO: 検出がない時も0値で逆伝播できるようにする
            # 検出がない場合は終了
            pil_adv_image = s3fd_util.image_decode(s3fd_adv_image[0], scale[0])
            tbx_writer.add_image(
                "adversarial_image",
                transforms.functional.to_tensor(pil_adv_image),
                perturbate_iter,
            )
            break

        tpc_loss, tps_loss, fpc_loss, end_flag = total_loss(
            detections,
            gt_detections,
            background_patch_boxes,
            s3fd_adv_background_image.shape[-2:],
            config.loss,
        )
        loss = tpc_loss + tps_loss + fpc_loss

        if end_flag:
            # 損失計算時に条件を満たした場合終了(zの要素がすべて0の場合)
            # 原因は特定していないがzの要素が全て0の場合勾配画像がバグるので
            pil_adv_image = s3fd_util.image_decode(s3fd_adv_image[0], scale[0])
            tbx_writer.add_image(
                "adversarial_image",
                transforms.functional.to_tensor(pil_adv_image),
                perturbate_iter,
            )
            break

        optimizer.zero_grad()
        loss.backward()

        gradient_image = s3fd_adv_background_image.grad

        with torch.no_grad():
            if perturbate_iter == 0:
                # ループの最初にのみ実行
                # パッチ領域を決定する
                # NOTE:十分な背景領域が存在しない場合、パッチは選択されない
                background_patch_boxes = pf.initial_background_patches(
                    gt_detections,
                    gradient_image,
                    config.initial_background_patches,
                    scale=bg_scale,
                )
            else:
                # パッチ領域を拡大する（縮小はしない）
                background_patch_boxes = pf.expanded_background_patches(
                    background_patch_boxes,
                    gt_detections,
                    gradient_image,
                    config.expanded_background_patches,
                    scale=bg_scale,
                )

            # 勾配画像をパッチ領域の形に切り出す
            perturbated_image = pf.perturbation_in_background_patches(
                gradient_image, background_patch_boxes * bg_scale
            )
            # make_box_image(perturbated_image, background_patch_boxes)
            # パッチの正規化
            nomalized_perturbated_image = pf.perturbation_normalization(
                perturbated_image, config.perturbation_normalization
            )
            # make_box_image(perturbated_image, background_patch_boxes)
            # adv_image-perturbated_imageの計算結果を[0,255]にクリップする
            s3fd_adv_background_image = pf.update_i_with_pixel_clipping(
                s3fd_adv_background_image, nomalized_perturbated_image
            )

            if tbx_writer is not None:
                tbx_writer.add_scalar("total_loss", loss, perturbate_iter)
                tbx_writer.add_scalar("tpc_loss", tpc_loss, perturbate_iter)
                tbx_writer.add_scalar("tps_loss", tps_loss, perturbate_iter)
                tbx_writer.add_scalar("fpc_loss", fpc_loss, perturbate_iter)

                bp_image = imgdraw.draw_boxes(
                    s3fd_util.image_decode(s3fd_adv_background_image[0]),
                    background_patch_boxes * bg_scale,
                )
                tbx_writer.add_image(
                    "background_patch_boxes",
                    transforms.functional.to_tensor(bp_image),
                    perturbate_iter,
                )

                if detections is not None:
                    pil_adv_image = s3fd_util.image_decode(s3fd_adv_image[0], scale[0])
                    det_image = imgdraw.draw_boxes(
                        pil_adv_image, detections.get_image_xyxy()
                    )
                    tbx_writer.add_image(
                        "adversarial_image",
                        transforms.functional.to_tensor(det_image),
                        perturbate_iter,
                    )

    adv_background_image = s3fd_util.image_decode(s3fd_adv_background_image[0])
    return adv_background_image


@hydra.main(
    version_base=None, config_path="../conf/", config_name="train_background_patch"
)
def main(cfg: DictConfig):
    config = cfg.train_main
    orig_wd_path = os.getcwd()

    if config.image_path == "":
        print("please select path")
        print(
            "Usage: python ssd-background-patches/eval_background.py path={path/to/adversarial_background_image}"
        )
        return

    image_path = os.path.join(orig_wd_path, config.image_path)
    # adv_bg_image = yolo_util.get_yolo_format_image_from_file(adv_bg_image_path)
    image = Image.open(image_path)

    mask_image_path = os.path.join(orig_wd_path, config.mask_image_path)
    mask_image = Image.open(mask_image_path)

    mode = config.mode

    match mode:
        case "monitor":
            tbx_writer = SummaryWriter(config.output_dir)

            with torch.autograd.detect_anomaly():
                adv_background_image = train_adversarial_image(
                    image,
                    mask_image,
                    config.train_adversarial_image,
                    tbx_writer=tbx_writer,
                )

            tbx_writer.close()

            output_image_path = os.path.join(config.output_dir, "adv_image.png")

            adv_background_image.save(output_image_path)
            print("finished!")

        case "evaluate":
            adv_background_image = train_adversarial_image(
                image, mask_image, config.train_adversarial_image
            )

            output_adv_path = os.path.join(
                config.output_dir, f"adv_background_patch_image.png"
            )

            adv_background_image.save(output_adv_path)

        case _:
            raise Exception("modeが想定されていない値です")


if __name__ == "__main__":
    main()
