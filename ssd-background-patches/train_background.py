import os

import hydra
from omegaconf import DictConfig

import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms.functional import resize, to_tensor, to_pil_image

import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm

from model.base_util import BaseTrainer
from box import boxio
from model import base_util, s3fd_util
from loss import proposed
from imageutil import imgseg, imgdraw


def train_adversarial_image(trainer: BaseTrainer, ground_truthes, config: DictConfig,  tbx_writer=None):
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

    # 敵対的背景
    # (1237,1649) is size of dataset image in S3FD representation
    s3fd_adv_background_image = torch.zeros((3, 1237, 1649), device=device)

    optimizer = optim.Adam([s3fd_adv_background_image])

    for epoch in tqdm(range(max_epoch)):
        epoch_loss_list = list()
        epoch_tpc_list = list()
        epoch_tps_list = list()
        epoch_fpc_list = list()
        # epoch_tv_list = list()

        for (image_list, mask_image_list), image_info in tqdm(image_loader, leave=False):

            scale_list = torch.cat([image_info['width'], image_info['height'],
                                    image_info['width'], image_info['height']]).T.to(device=device, dtype=torch.float)

            # Preprocessing
            # Set to no_grad since the process is not needed for gradient calculation.
            with torch.no_grad():

                image_list = image_list.to(
                    device=device, dtype=torch.float) - trainer.get
                mask_image_list = mask_image_list.to(
                    device=device)

            s3fd_adv_background_image.requires_grad = True
            s3fd_adv_image_list = imgseg.composite_image(
                image_list, s3fd_adv_background_image, mask_image_list)

            # Detection from adversarial images
            adv_output = model(s3fd_adv_image_list)
            adv_detections_list = trainer.make_detections_list(
                adv_output, scale_list, s3fd_util.detections_s3fd_loss, config.model_thresh)

            tpc_loss_list = torch.zeros(
                image_loader.batch_size, device=device)
            tps_loss_list = torch.zeros(
                image_loader.batch_size, device=device)
            fpc_loss_list = torch.zeros(
                image_loader.batch_size, device=device)
            # tv_loss_list = torch.zeros(image_loader.batch_size, device=device)

            for i in range(image_loader.batch_size):
                if adv_detections_list[i] is None:
                    tpc_loss_list[i] += 0
                    tps_loss_list[i] += 0
                    fpc_loss_list[i] += 0
                    tv_loss_list[i] += 0
                    continue

                tpc_loss, tps_loss, fpc_loss, tv_loss = proposed.total_loss(
                    adv_detections_list[i], ground_truthes, s3fd_adv_background_image.unsqueeze(0), config.loss)

                tpc_loss_list[i] += tpc_loss
                tps_loss_list[i] += tps_loss
                fpc_loss_list[i] += fpc_loss
                # tv_loss_list[i] += tv_loss

            mean_tpc = torch.mean(tpc_loss_list)
            mean_tps = torch.mean(tps_loss_list)
            mean_fpc = torch.mean(fpc_loss_list)
            # mean_tv = torch.mean(tv_loss_list)

            loss = mean_tpc+mean_tps+mean_fpc  # +mean_tv

            if loss == 0:
                continue

            optimizer.zero_grad()
            loss.backward()
            # The Adversarial background image is updated here
            optimizer.step()

            with torch.no_grad():

                # tensorboard
                epoch_loss_list.append(loss.detach(
                ).cpu().resolve_conj().resolve_neg().numpy())
                epoch_tpc_list.append(mean_tpc.detach(
                ).cpu().resolve_conj().resolve_neg().numpy())
                epoch_tps_list.append(mean_tps.detach(
                ).cpu().resolve_conj().resolve_neg().numpy())
                epoch_fpc_list.append(mean_fpc.detach(
                ).cpu().resolve_conj().resolve_neg().numpy())
                # epoch_tv_list.append(mean_tv.detach().cpu().resolve_conj().resolve_neg().numpy())

        with torch.no_grad():
            # tensorboard
            epoch_mean_loss = np.array(epoch_loss_list).mean()
            epoch_mean_tpc = np.array(epoch_tpc_list).mean()
            epoch_mean_tps = np.array(epoch_tps_list).mean()
            epoch_mean_fpc = np.array(epoch_fpc_list).mean()
            # epoch_mean_tv = np.array(epoch_tv_list).mean()

            if tbx_writer is not None:
                tbx_writer.add_scalar(
                    "total_loss", epoch_mean_loss, epoch)
                tbx_writer.add_scalar(
                    "tpc_loss", epoch_mean_tpc, epoch)
                tbx_writer.add_scalar(
                    "tps_loss", epoch_mean_tps, epoch)
                tbx_writer.add_scalar(
                    "fpc_loss", epoch_mean_fpc, epoch)
                # tbx_writer.add_scalar("tv_loss", epoch_mean_tv, epoch)

                tbx_writer.add_image(
                    "adversarial_background_image", s3fd_adv_background_image, epoch)

                if adv_detections_list[0] is not None:
                    tbx_anno_adv_image = to_tensor(imgdraw.draw_boxes(
                        resize(s3fd_adv_image_list[0], scale_list[0]).to_pil_image(), adv_detections_list[0].get_image_xyxy()))
                    tbx_writer.add_image(
                        "adversarial_image", tbx_anno_adv_image, epoch)
                else:
                    tbx_anno_adv_image = resize(
                        s3fd_adv_image_list[0], (image_info['height'][0], image_info['width'][0]))
                    tbx_writer.add_image(
                        "adversarial_image", tbx_anno_adv_image, epoch)
    return s3fd_adv_background_image.clone().cpu()


@ hydra.main(version_base=None, config_path="../conf/", config_name="train_background")
def main(cfg: DictConfig):
    config = cfg.train_main
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mode = config.mode

    match mode:
        case "monitor":
            mode_config = config.monitor_mode

        case "evaluate":
            mode_config = config.evaluate_mode

        case _:
            raise Exception('modeが想定されていない値です')

    gt_conf_list, gt_box_list = boxio.generate_integrated_xyxy_list(
        mode_config.dataset_factory.detection_path, max_iter=mode_config.dataset_factory.max_iter)
    # TODO:be dynamic
    ground_truthes = s3fd_util.detections_base(
        gt_conf_list.to(device=device), gt_box_list.to(device=device), is_xywh=False)

    tbx_writer = SummaryWriter(config.output_dir)

    trainer: BaseTrainer = hydra.utils.instantiate(
        mode_config)

    adv_background_image = train_adversarial_image(
        trainer, ground_truthes, config.train_adversarial_image, tbx_writer=tbx_writer)

    tbx_writer.close()

    output_adv_path = os.path.join(
        config.output_dir, "adv_background_image.png")

    pil_image = s3fd_util.image_decode(adv_background_image)
    pil_image.save(output_adv_path)
    print("finished!")


if __name__ == "__main__":
    main()
