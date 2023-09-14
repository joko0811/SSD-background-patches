import os
import sys
import logging

import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig

import torch
import torch.optim as optim
from torchvision import transforms

import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm

from box import boxio
from box.boxconv import xyxy2xywh
from model.S3FD.layers.modules.multibox_loss import MultiBoxLoss

from detection.detection_base import DetectionsBase
from model.base_util import BackgroundBaseTrainer
from patchutil.positional_patch import PositionalBackgroundManager
from util.infoutil import get_git_sha


def train_adversarial_image(
    trainer: BackgroundBaseTrainer,
    background_manager: PositionalBackgroundManager,
    ground_truthes: DetectionsBase,
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

    lr = 0.1
    lr_decay = 0.95
    lr_decay_epoch = 5
    momentum = 0.9

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_loader = trainer.get_dataloader()
    model = trainer.load_model(mode="train")
    model.eval()

    min_loss = sys.maxsize

    # 敵対的背景
    # (1237,1649) is size of dataset image in S3FD representation
    # s3fd_dataset_image_format = (3, 1237, 1649)
    # retina_dataset_image_format = (3, 840, 840)

    adv_patch = background_manager.generate_patch().to(device)
    patch_size = adv_patch.shape[1:]
    torch.manual_seed(0)
    adv_patch = torch.rand(adv_patch.shape, device=device) * 255
    patch_coordinate = (0, 0)

    patch_dir = os.path.join(config.output_dir, "patch")
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)

    multibox_conf = {
        "NUM_CLASSES": 2,
        "NEG_POS_RATIOS": 3,
        "VARIANCE": [0.1, 0.2],
        "FACE": {"OVERLAP_THRESH": [0.1, 0.35, 0.5]},
    }
    multibox_conf = OmegaConf.create(multibox_conf)
    multibox_loss = MultiBoxLoss(multibox_conf, dataset="face", use_gpu=True)

    """
    optimizer = torch.optim.SGD(
        [adv_patch],
        lr=lr,
        momentum=momentum,
    )
    """

    for epoch in tqdm(range(max_epoch)):
        epoch_loss_list = list()
        epoch_l_list = list()
        epoch_c_list = list()

        for (image_list, mask_list), image_info in image_loader:
            # Preprocessing
            # Set to no_grad since the process is not needed for gradient calculation.
            with torch.no_grad():
                image_list = image_list.to(device=device, dtype=torch.float)
                mask_list = mask_list.to(device=device)

            # if adv_patch.grad is not None: adv_patch.grad.zero_()
            adv_patch.requires_grad = True

            image_size = image_list[0].shape[1:]  # (H,W)

            args_of_tpatch = background_manager.generate_kwargs_of_transform_patch(
                image_size, patch_size, xyxy2xywh(image_info["xyxy"])[:, :, 2:]
            )
            (
                adv_background_image,
                adv_background_mask,
            ) = background_manager.transform_patch(
                adv_patch, image_size, **args_of_tpatch
            )

            adv_image_list = background_manager.apply(
                adv_background_image, adv_background_mask, image_list, mask_list
            )

            # Detection from adversarial images
            adv_output = model(adv_image_list)

            l_loss_list = torch.zeros(image_loader.batch_size, device=device)
            c_loss_list = torch.zeros(image_loader.batch_size, device=device)

            for i in range(image_loader.batch_size):
                # shape: [batch_size,num_objs,5]
                # 5: [x1,y1,x2,y2,score]
                if image_info["xyxy"][i].nelement() == 0:
                    l_loss_list[i] += 0
                    c_loss_list[i] += 0
                    continue

                target = [
                    torch.cat(
                        [
                            image_info["xyxy"][i],
                            torch.tensor([1]).tile(image_info["xyxy"][i].shape[0], 1),
                        ],
                        dim=1,
                    ).to(device=device, dtype=torch.float)
                ]
                loss_l, loss_c = multibox_loss(adv_output, target)

                l_loss_list[i] += loss_l
                c_loss_list[i] += loss_c

            mean_l = torch.mean(l_loss_list)
            mean_c = torch.mean(c_loss_list)

            loss = mean_l + mean_c

            with torch.no_grad():
                # tensorboard
                epoch_loss_list.append(
                    loss.detach().cpu().resolve_conj().resolve_neg().numpy()
                )
                epoch_l_list.append(
                    mean_l.detach().cpu().resolve_conj().resolve_neg().numpy()
                )
                epoch_c_list.append(
                    mean_c.detach().cpu().resolve_conj().resolve_neg().numpy()
                )
                # epoch_tv_list.append(mean_tv.detach().cpu().resolve_conj().resolve_neg().numpy())

            if loss == 0:
                continue

            # update the patch
            loss.backward()
            grad = adv_patch.grad

            with torch.no_grad():
                # normalize gradients by dividing l_infinity norm
                grad_linf = grad.detach().abs().max()
                if grad_linf > 0:
                    norm_grad = (grad / grad_linf) * 255
                else:
                    norm_grad = grad
                """
                optimizer.step()
                optimizer.zero_grad()
                """

                # grad_sign = norm_grad.detach().sign()
                # adv_patch += lr * grad_sign
                adv_patch = adv_patch + lr * norm_grad

                adv_patch = adv_patch.clamp(0.0, 255.0)
                # adv_patch.grad.zero_()

        with torch.no_grad():
            if epoch % lr_decay_epoch == 0:
                lr = lr * lr_decay

            logging.info("epoch: " + str(epoch))
            # tensorboard
            epoch_mean_loss = np.array(epoch_loss_list).mean()
            epoch_mean_l = np.array(epoch_l_list).mean()
            epoch_mean_c = np.array(epoch_c_list).mean()

            logging.info("epoch_mean_loss: " + str(epoch_mean_loss))

            torch.save(
                adv_patch,
                os.path.join(patch_dir, "epoch" + str(epoch) + "_patch.pt"),
            )

            if tbx_writer is not None:
                tbx_writer.add_scalar("total_loss", epoch_mean_loss, epoch)
                tbx_writer.add_scalar("loss_l", epoch_mean_l, epoch)
                tbx_writer.add_scalar("loss_c", epoch_mean_c, epoch)

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
                    tbx_anno_adv_image = transforms.functional.to_tensor(
                        trainer.transformed2pil(
                            adv_image_list[0],
                            (image_info["height"][0], image_info["width"][0]),
                        )
                    )
                    tbx_writer.add_image("adversarial_image", tbx_anno_adv_image, epoch)
            # エポック毎のバッファリングのフラッシュ
            sys.stdout.flush()
    return adv_patch.clone().cpu()


@hydra.main(version_base=None, config_path="../conf/", config_name="train_background")
def main(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tbx_writer = SummaryWriter(cfg.output_dir)
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
    background_manager = PositionalBackgroundManager()

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
    logging.info("finished!")


if __name__ == "__main__":
    main()
