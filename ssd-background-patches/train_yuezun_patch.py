import os
import sys
import logging

import hydra
from omegaconf import DictConfig
import torch
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

from imageutil import imgdraw
from model.base_util import BackgroundBaseTrainer
from detection.tp_fp_manager import TpFpManager
from util.infoutil import get_git_sha
from util.clustering import object_grouping
from loss.li2019 import total_loss
import proposed_func as pf


@hydra.main(version_base=None, config_path="../conf/", config_name="train_yuezun_patch")
def train_adversarial_image(cfg: DictConfig):
    device = torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available() else "cpu")

    mode = cfg.mode

    logging.info("device: " + str(device))
    git_hash = get_git_sha()
    logging.info("commit hash: " + git_hash)

    match mode:
        case "monitor":
            mode_trainer = cfg.monitor_trainer

        case "evaluate":
            mode_trainer = cfg.evaluate_trainer

        case _:
            raise Exception("modeが想定されていない値です")

    trainer: BackgroundBaseTrainer = hydra.utils.instantiate(mode_trainer)

    max_epoch = cfg.train_parameters.max_epoch  # default 250

    image_loader = trainer.get_dataloader()
    model = trainer.load_model(device=device, mode="test")
    model.to(device=device)
    model.eval()

    patch_dir = os.path.join(cfg.output_dir, "patch")
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)

    for image_list, target in tqdm(image_loader):

        tbx_writer = SummaryWriter(cfg.output_dir)

        tp_fp_manager = TpFpManager(device=device)

        # Preprocessing
        # Set to no_grad since the process is not needed for gradient calculation.
        with torch.no_grad():
            image_list = image_list.to(device=device, dtype=torch.float)

            gt_output = model(image_list)

            ground_truthes = trainer.make_detections_list(gt_output)

            # gtをクラスタリング、群を作る
            group_labels = torch.from_numpy(
                object_grouping(ground_truthes[0].xywh.detach().cpu().numpy())
            ).to(device=device, dtype=torch.float)
            total_group = len(torch.unique(group_labels))

            # 1つの群に対して担当する領域数を定める
            patch_num_for_group = cfg.train_parameters.patch_num_for_group
            # 上記領域の初期位置
            background_patch_boxes = torch.zeros(
                total_group * patch_num_for_group, 4
            ).to(device=device)
            image_size = image_list[0].shape[1:]  # (H,W)

        for epoch in range(max_epoch):

            image_list.requires_grad = True

            # Detection from adversarial images
            adv_output = model(image_list)
            adv_detections_list = trainer.make_detections_list(
                adv_output,
                cfg.train_parameters.model_thresh,
                image_list.shape[1:],
                [list(image_size)],
            )

            with torch.no_grad():
                bg_scale = torch.tensor(
                    [image_size[1], image_size[0], image_size[1], image_size[0]]
                ).to(device=device)
                tp_fp_manager.add_detection(adv_detections_list[0], ground_truthes[0])

            tpc_loss, tps_loss, fpc_loss, end_flag = total_loss(
                adv_detections_list[0],
                ground_truthes[0],
                background_patch_boxes,
                image_list[0].shape[-2:],
                cfg.loss,
            )
            loss = tpc_loss + tps_loss + fpc_loss
            if loss == 0:
                continue

            image_list[0].grad = None
            loss.backward()
            gradient_image = image_list[0].grad
            # The Adversarial background image is updated here

            with torch.no_grad():
                if epoch == 0:
                    background_patch_boxes = pf.initial_background_patches(
                        ground_truthes[0], gradient_image, cfg, bg_scale
                    )
                else:
                    background_patch_boxes = pf.expanded_background_patches(
                        background_patch_boxes,
                        ground_truthes[0],
                        gradient_image,
                        cfg,
                        bg_scale,
                    )
                perturbated_image = pf.perturbation_in_background_patches(
                    gradient_image, background_patch_boxes * bg_scale
                )
                nomalized_perturbated_image = pf.perturbation_normalization(
                    perturbated_image, cfg.perturbation_normalization
                )
                # make_box_image(perturbated_image, background_patch_boxes)
                # adv_image-perturbated_imageの計算結果を[0,255]にクリップする
                s3fd_adv_background_image = pf.update_i_with_pixel_clipping(
                    s3fd_adv_background_image, nomalized_perturbated_image
                )

    with torch.no_grad():
        tp, fp, fn, gt = tp_fp_manager.get_value()

        logging.info("epoch: " + str(epoch))

        if tbx_writer is not None:
            tbx_writer.add_image(
                "adversarial_background_image",
                transforms.functional.pil_to_tensor(
                    trainer.transformed2pil(image_list[0])
                ),
                epoch,
            )

            if epoch % 10 == 0:
                if adv_detections_list[0] is not None:
                    tbx_anno_adv_image = transforms.functional.pil_to_tensor(
                        imgdraw.draw_boxes(
                            trainer.transformed2pil(
                                image_list[0], image_list[0].shape[-2:]
                            ),
                            adv_detections_list[0].xyxy * image_list.shape[1:],
                        )
                    )
                    tbx_writer.add_image("adversarial_image", tbx_anno_adv_image, epoch)
                else:
                    tbx_anno_adv_image = transforms.functional.pil_to_tensor(
                        trainer.transformed2pil(
                            image_list[0],
                            image_list[0].shape[-2:],
                        )
                    )
                    tbx_writer.add_image("adversarial_image", tbx_anno_adv_image, epoch)
        # エポック毎のバッファリングのフラッシュ
        sys.stdout.flush()


def main():
    train_adversarial_image()


if __name__ == "__main__":
    main()
