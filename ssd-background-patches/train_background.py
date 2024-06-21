import os
import sys
import logging

import hydra
from omegaconf import DictConfig

import torch
from torchvision import transforms

from tensorboardX import SummaryWriter
from tqdm import tqdm

from box import boxio
from box.boxconv import xyxy2xywh
from loss.loss_calculator_recorder import LossCalculatorRecorder
from imageutil import imgdraw

from detection.detection_base import DetectionsBase
from model.base_util import BackgroundBaseTrainer
from ptmanager.base_patch import BaseBackgroundManager
from detection.tp_fp_manager import TpFpManager
from util.infoutil import get_git_sha


@hydra.main(version_base=None, config_path="../conf/", config_name="train_background")
def train_adversarial_image(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    background_manager: BaseBackgroundManager = hydra.utils.instantiate(cfg.ptmanager)

    # 全ての正しい検出の読み取り・生成
    all_gt_conf_list, all_gt_box_list = boxio.generate_integrated_xyxy_list(
        mode_trainer.dataset_factory.detection_path,
        max_iter=mode_trainer.dataset_factory.max_iter,
    )
    all_ground_truthes = DetectionsBase(
        all_gt_conf_list.to(device), all_gt_box_list.to(device), is_xywh=False
    )

    max_epoch = cfg.train_parameters.max_epoch  # default 250

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_loader = trainer.get_dataloader()
    model = trainer.load_model(mode="test")
    model.eval()

    # 敵対的背景
    # (1237,1649) is size of dataset image in S3FD representation
    # s3fd_dataset_image_format = (3, 1237, 1649)
    # retina_dataset_image_format = (3, 840, 840)

    adv_patch = background_manager.generate_patch().to(device)
    patch_size = adv_patch.shape[1:]

    patch_dir = os.path.join(cfg.output_dir, "patch")
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)

    optimizer_factory = hydra.utils.instantiate(cfg.optim.optimizer)
    optimizer = optimizer_factory(params=[adv_patch])

    with SummaryWriter(cfg.train_parameters.output_dir) as tbx_writer:
        loss_calculator_recorder = LossCalculatorRecorder(
            cfg.loss, tbx_writer=tbx_writer
        )
        for epoch in tqdm(range(max_epoch)):
            tp_fp_manager = TpFpManager(device=device)

            for (image_list, mask_list), image_info in image_loader:
                # Preprocessing
                # Set to no_grad since the process is not needed for gradient calculation.
                with torch.no_grad():
                    image_list = image_list.to(device=device, dtype=torch.float)
                    mask_list = mask_list.to(device=device)
                    scale_list = torch.cat(
                        [
                            image_info["width"],
                            image_info["height"],
                            image_info["width"],
                            image_info["height"],
                        ]
                    ).T.to(device=device, dtype=torch.float)

                    ground_truthes = DetectionsBase(
                        conf_list=image_info["conf"]
                        .squeeze(0)
                        .to(device=device, dtype=torch.float),
                        box_list=image_info["xyxy"]
                        .squeeze(0)
                        .to(device=device, dtype=torch.float),
                        is_xywh=False,
                    )

                loss_calculator_recorder.init_per_iter()

                adv_patch.requires_grad = True

                image_size = image_list[0].shape[1:]  # (H,W)

                args_of_tpatch = background_manager.generate_kwargs_of_transform_patch(
                    image_size, patch_size, xyxy2xywh(image_info["xyxy"])[:, :, 2:]
                )
                (
                    tmp_adv_background_image,
                    adv_background_mask,
                ) = background_manager.transform_patch(
                    torch.clamp(adv_patch, min=0, max=255) / 255,
                    image_size,
                    **args_of_tpatch,
                )
                adv_background_image = tmp_adv_background_image * 255

                adv_image_list = background_manager.apply(
                    adv_background_image,
                    adv_background_mask,
                    image_list,
                    mask_list,
                )

                # Detection from adversarial images
                adv_output = model(adv_image_list)
                adv_detections_list = trainer.make_detections_list(
                    adv_output,
                    cfg.train_parameters.model_thresh,
                    scale_list,
                    [list(image_size)],
                )

                for i in range(image_loader.batch_size):
                    with torch.no_grad():
                        # 正しい顔領域を読み込んで追加する
                        if (
                            image_info["conf"][i].nelement() != 0
                            and image_info["xyxy"][i].nelement() != 0
                        ):
                            image_ground_truth = DetectionsBase(
                                image_info["conf"][i].to(device),
                                image_info["xyxy"][i].to(device),
                                is_xywh=False,
                            )
                        else:
                            image_ground_truth = None

                        tp_fp_manager.add_detection(
                            adv_detections_list[i], image_ground_truth
                        )

                    loss_calculator_recorder.step_per_img(
                        adv_detections_list[i], ground_truthes, all_ground_truthes
                    )

                loss = loss_calculator_recorder.step_per_iter()

                if loss == 0:
                    continue

                if hasattr(optimizer, "pre_backward"):
                    optimizer.pre_backward([adv_patch])

                optimizer.zero_grad()
                loss.backward()
                # The Adversarial background image is updated here
                optimizer.step()

            with torch.no_grad():
                loss_calculator_recorder.step_per_epoch(epoch)
                tp, fp, fn, gt = tp_fp_manager.get_value()

                logging.info("epoch: " + str(epoch))
                background_manager.save_best_image(
                    adv_patch,
                    os.path.join(patch_dir, "epoch" + str(epoch) + "_patch.pt"),
                    len(all_ground_truthes),
                    tp,
                    fp,
                    fn,
                )

                if tbx_writer is not None:
                    tbx_writer.add_image(
                        "adversarial_background_image",
                        transforms.functional.pil_to_tensor(
                            trainer.transformed2pil(adv_background_image)
                        ),
                        epoch,
                    )

                    if epoch % 10 == 0:
                        if adv_detections_list[0] is not None:
                            tbx_anno_adv_image = transforms.functional.pil_to_tensor(
                                imgdraw.draw_boxes(
                                    trainer.transformed2pil(
                                        adv_image_list[0],
                                        (
                                            image_info["height"][0],
                                            image_info["width"][0],
                                        ),
                                    ),
                                    adv_detections_list[0].xyxy * scale_list[0],
                                )
                            )
                            tbx_writer.add_image(
                                "adversarial_image", tbx_anno_adv_image, epoch
                            )
                        else:
                            tbx_anno_adv_image = transforms.functional.pil_to_tensor(
                                trainer.transformed2pil(
                                    adv_image_list[0],
                                    (image_info["height"][0], image_info["width"][0]),
                                )
                            )
                            tbx_writer.add_image(
                                "adversarial_image", tbx_anno_adv_image, epoch
                            )
                # エポック毎のバッファリングのフラッシュ
                sys.stdout.flush()

    output_adv_path = os.path.join(cfg.output_dir, "adv_background_image.png")
    pil_image = transforms.functional.to_pil_image(
        torch.clamp(adv_patch, min=0, max=255).trunc().clone().cpu() / 255
    )
    pil_image.save(output_adv_path)
    logging.info("finished!")
    return loss


def main():
    train_adversarial_image()


if __name__ == "__main__":
    main()
