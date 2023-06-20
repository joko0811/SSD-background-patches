import os

import hydra
from omegaconf import DictConfig

import torch
import torch.optim as optim
from torchvision import transforms

import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm

from box import boxio
from loss import simple
from imageutil import imgdraw

from detection.detection_base import DetectionsBase
from model.base_util import BackgroundBaseTrainer
from patch_manager import BaseBackgroundManager
from detection.tp_fp_manager import TpFpManager


def train_adversarial_image(
    trainer: BackgroundBaseTrainer,
    background_manager: BaseBackgroundManager,
    ground_truthes,
    config: DictConfig,
    tbx_writer=None,
):
    """
    Args:
        model: S3FD
        image_loader: DataLoader with dataset.train_background.TrainBackgroundDataset
        ground_truthes: A two-dimensional list summarizing all image detections in the Dataset.(X*4)
        config: conf.train_background.train_adversarial_image
    Return:
        Tensor Image of adversarial background

    """

    max_epoch = config.max_epoch  # default 250

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_loader = trainer.get_dataloader()
    model = trainer.load_model()
    model.eval()

    tp_fp_manager = TpFpManager(ground_truthes)

    # 敵対的背景
    # (1237,1649) is size of dataset image in S3FD representation
    # s3fd_dataset_image_format = (3, 1237, 1649)
    # retina_dataset_image_format = (3, 840, 840)

    adv_patch = background_manager.generate_patch().to(device)
    optimizer = optim.Adam([adv_patch])

    for epoch in tqdm(range(max_epoch)):
        epoch_loss_list = list()
        epoch_tpc_list = list()
        # epoch_tps_list = list()
        epoch_fpc_list = list()
        # epoch_tv_list = list()
        tp_fp_manager.reset()

        for (image_list, mask_list), image_info in image_loader:
            scale_list = torch.cat(
                [
                    image_info["width"],
                    image_info["height"],
                    image_info["width"],
                    image_info["height"],
                ]
            ).T.to(device=device, dtype=torch.float)

            # Preprocessing
            # Set to no_grad since the process is not needed for gradient calculation.
            with torch.no_grad():
                image_list = image_list.to(device=device, dtype=torch.float)
                mask_list = mask_list.to(device=device)

            adv_patch.requires_grad = True
            adv_background_image = background_manager.transform_patch(adv_patch)
            adv_image_list = background_manager.apply(
                adv_background_image, image_list, mask_list
            )

            # Detection from adversarial images
            adv_output = model(adv_image_list)
            adv_detections_list = trainer.make_detections_list(
                adv_output, config.model_thresh
            )

            tpc_loss_list = torch.zeros(image_loader.batch_size, device=device)
            # tps_loss_list = torch.zeros(
            #    image_loader.batch_size, device=device)
            fpc_loss_list = torch.zeros(image_loader.batch_size, device=device)
            # tv_loss_list = torch.zeros(image_loader.batch_size, device=device)

            for i in range(image_loader.batch_size):
                with torch.no_grad():
                    tp_fp_manager.add_detection(adv_detections_list[i])

                if adv_detections_list[i] is None:
                    tpc_loss_list[i] += 0
                    # tps_loss_list[i] += 0
                    fpc_loss_list[i] += 0
                    # tv_loss_list[i] += 0
                    continue

                # tpc_loss, tps_loss, fpc_loss = proposed.total_loss(
                #     adv_detections_list[i], ground_truthes, adv_background_image.unsqueeze(0), config.loss, scale=scale_list[0])
                tpc_loss, fpc_loss = simple.total_loss(
                    adv_detections_list[i], ground_truthes, config.loss
                )

                tpc_loss_list[i] += tpc_loss
                # tps_loss_list[i] += tps_loss
                fpc_loss_list[i] += fpc_loss
                # tv_loss_list[i] += tv_loss

            mean_tpc = torch.mean(tpc_loss_list)
            # mean_tps = torch.mean(tps_loss_list)
            mean_fpc = torch.mean(fpc_loss_list)
            # mean_tv = torch.mean(tv_loss_list)

            # loss = mean_tpc+mean_tps+mean_fpc  # +mean_tv
            loss = mean_tpc + mean_fpc

            if loss == 0:
                continue

            optimizer.zero_grad()
            loss.backward()
            # The Adversarial background image is updated here
            optimizer.step()

            with torch.no_grad():
                # tensorboard
                epoch_loss_list.append(
                    loss.detach().cpu().resolve_conj().resolve_neg().numpy()
                )
                epoch_tpc_list.append(
                    mean_tpc.detach().cpu().resolve_conj().resolve_neg().numpy()
                )
                # epoch_tps_list.append(mean_tps.detach(
                # ).cpu().resolve_conj().resolve_neg().numpy())
                epoch_fpc_list.append(
                    mean_fpc.detach().cpu().resolve_conj().resolve_neg().numpy()
                )
                # epoch_tv_list.append(mean_tv.detach().cpu().resolve_conj().resolve_neg().numpy())

        with torch.no_grad():
            tp, fp, gt = tp_fp_manager.get_value()
            print("epoch: " + str(epoch))
            background_manager.save_best_image(
                adv_patch,
                os.path.join(config.output_dir, "epoch" + str(epoch) + "_patch.pt"),
                ground_truthes,
                tp,
                fp,
            )
            # tensorboard
            epoch_mean_loss = np.array(epoch_loss_list).mean()
            epoch_mean_tpc = np.array(epoch_tpc_list).mean()
            # epoch_mean_tps = np.array(epoch_tps_list).mean()
            epoch_mean_fpc = np.array(epoch_fpc_list).mean()
            # epoch_mean_tv = np.array(epoch_tv_list).mean()

            if tbx_writer is not None:
                tbx_writer.add_scalar("total_loss", epoch_mean_loss, epoch)
                tbx_writer.add_scalar("tpc_loss", epoch_mean_tpc, epoch)
                # tbx_writer.add_scalar(
                #     "tps_loss", epoch_mean_tps, epoch)
                tbx_writer.add_scalar("fpc_loss", epoch_mean_fpc, epoch)
                # tbx_writer.add_scalar("tv_loss", epoch_mean_tv, epoch)

                tbx_writer.add_image(
                    "adversarial_background_image",
                    transforms.functional.to_tensor(
                        trainer.transformed2pil(
                            adv_background_image, trainer.get_image_size()
                        )
                    ),
                    epoch,
                )

                if epoch % 10 == 0:
                    if adv_detections_list[0] is not None:
                        tbx_anno_adv_image = transforms.functional.to_tensor(
                            imgdraw.draw_boxes(
                                trainer.transformed2pil(
                                    adv_image_list[0],
                                    (image_info["height"][0], image_info["width"][0]),
                                ),
                                adv_detections_list[0].xyxy * scale_list[0],
                            )
                        )
                        tbx_writer.add_image(
                            "adversarial_image", tbx_anno_adv_image, epoch
                        )
                    else:
                        tbx_anno_adv_image = transforms.functional.to_tensor(
                            trainer.transformed2pil(
                                adv_image_list[0],
                                (image_info["height"][0], image_info["width"][0]),
                            )
                        )
                        tbx_writer.add_image(
                            "adversarial_image", tbx_anno_adv_image, epoch
                        )
    return adv_patch.clone().cpu()


@hydra.main(version_base=None, config_path="../conf/", config_name="train_background")
def main(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tbx_writer = SummaryWriter(cfg.output_dir)
    mode = cfg.mode

    match mode:
        case "monitor":
            mode_trainer = cfg.monitor_trainer

        case "evaluate":
            mode_trainer = cfg.evaluate_trainer

        case _:
            raise Exception("modeが想定されていない値です")

    trainer: BackgroundBaseTrainer = hydra.utils.instantiate(mode_trainer)
    background_manager: BaseBackgroundManager = hydra.utils.instantiate(
        cfg.patch_manager
    )(trainer.get_image_size(), mode="test")

    # 全ての正しい検出の読み取り・生成
    gt_conf_list, gt_box_list = boxio.generate_integrated_xyxy_list(
        mode_trainer.dataset_factory.detection_path,
        max_iter=mode_trainer.dataset_factory.max_iter,
    )
    ground_truthes = DetectionsBase(
        gt_conf_list.to(device), gt_box_list.to(device), is_xywh=False
    )

    adv_background_image = train_adversarial_image(
        trainer,
        background_manager,
        ground_truthes,
        cfg.train_adversarial_image,
        tbx_writer=tbx_writer,
    )

    tbx_writer.close()

    output_adv_path = os.path.join(cfg.output_dir, "adv_background_image.png")

    pil_image = transforms.functional.to_pil_image(adv_background_image)
    pil_image.save(output_adv_path)
    print("finished!")


if __name__ == "__main__":
    main()
