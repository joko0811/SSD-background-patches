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
from imageutil import imgconv
from model import s3fd_util
from dataset.mask import DirectoryImageWithMaskDataset
from loss import proposed
from imageutil import imgseg, imgdraw


def train_adversarial_image(model, image_loader, ground_truthes, config: DictConfig,  tbx_writer=None):

    max_epoch = config.max_epoch  # default 250

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 敵対的背景
    # adv_background_image = torch.zeros((3, 416, 416), device=device)
    # (1237,1649) is size of dataset image in S3FD representation
    s3fd_adv_background_image = torch.zeros((3, 1237, 1649), device=device)

    optimizer = optim.Adam([s3fd_adv_background_image])

    for epoch in tqdm(range(max_epoch)):
        epoch_loss_list = list()
        epoch_tpc_list = list()
        epoch_tps_list = list()
        epoch_fpc_list = list()
        epoch_tv_list = list()

        for image_list, mask_image_list, image_path_list, mask_image_path_list in tqdm(image_loader, leave=False):

            # Preprocessing
            # Set to no_grad since the process is not needed for gradient calculation.
            with torch.no_grad():
                """
                gpu_image_list = image_list.to(
                    device=device, dtype=torch.float)
                gpu_mask_image_list = mask_image_list.to(
                    device=device, dtype=torch.float)

                # Detection from unprocessed images
                # Detection of unprocessed images as Groundtruth
                gt_output = model(gpu_image_list)
                gt_nms_out = yolo_util.nms(gt_output)
                gt_detections_list = yolo_util.make_detections_list(
                    gt_nms_out, yolo_util.detections_yolo)
                """
                pil_image_list = imgconv.tensor2pil(image_list)
                encoded_tuple = s3fd_util.image_list_encode(
                    pil_image_list)
                s3fd_image_list = encoded_tuple[0].to(
                    device=device, dtype=torch.float)
                scale_list = encoded_tuple[1].to(
                    device=device, dtype=torch.float)

                pil_mask_image_list = imgconv.tensor2pil(mask_image_list)
                s3fd_mask_image_list = s3fd_util.image_list_encode(
                    pil_mask_image_list, is_mask=True)[0].to(device=device, dtype=torch.float)

            s3fd_adv_background_image.requires_grad = True
            # adv_image_list = bgutil.background_applyer(gpu_image_list, adv_background_image)
            s3fd_adv_image_list = imgseg.composite_image(
                s3fd_image_list, s3fd_adv_background_image, s3fd_mask_image_list)

            """
            # Detection from adversarial images
            adv_output = model(adv_image_list)
            adv_nms_out = yolo_util.nms(adv_output)
            adv_detections_list = yolo_util.make_detections_list(
                adv_nms_out, yolo_util.detections_yolo_loss)
            """
            # Detection from adversarial images
            adv_output = model(s3fd_adv_image_list)
            adv_detections_list = s3fd_util.make_detections_list(
                adv_output, scale_list, s3fd_util.detections_s3fd_loss, config.model_thresh)

            tpc_loss_list = torch.zeros(
                image_loader.batch_size, device=device)
            tps_loss_list = torch.zeros(
                image_loader.batch_size, device=device)
            fpc_loss_list = torch.zeros(
                image_loader.batch_size, device=device)
            tv_loss_list = torch.zeros(
                image_loader.batch_size, device=device)

            for i in range(image_loader.batch_size):
                if adv_detections_list[i] is None:
                    tpc_loss_list[i] += 0
                    tps_loss_list[i] += 0
                    fpc_loss_list[i] += 0
                    tv_loss_list[i] += 0
                    continue

                tpc_loss, tps_loss, fpc_loss, tv_loss = proposed.total_loss(
                    adv_detections_list[i], ground_truthes, s3fd_adv_background_image, config.loss)

                tpc_loss_list[i] += tpc_loss
                tps_loss_list[i] += tps_loss
                fpc_loss_list[i] += fpc_loss
                tv_loss_list[i] += tv_loss

            mean_tpc = torch.mean(tpc_loss_list)
            mean_tps = torch.mean(tps_loss_list)
            mean_fpc = torch.mean(fpc_loss_list)
            mean_tv = torch.mean(tv_loss_list)

            loss = mean_tpc+mean_tps+mean_fpc+mean_tv

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
                epoch_tv_list.append(mean_tv.detach(
                ).cpu().resolve_conj().resolve_neg().numpy())

        with torch.no_grad():
            # tensorboard
            epoch_mean_loss = np.array(epoch_loss_list).mean()
            epoch_mean_tpc = np.array(epoch_tpc_list).mean()
            epoch_mean_tps = np.array(epoch_tps_list).mean()
            epoch_mean_fpc = np.array(epoch_fpc_list).mean()
            epoch_mean_tv = np.array(epoch_tv_list).mean()

            if tbx_writer is not None:
                tbx_writer.add_scalar(
                    "total_loss", epoch_mean_loss, epoch)
                tbx_writer.add_scalar(
                    "tpc_loss", epoch_mean_tpc, epoch)
                tbx_writer.add_scalar(
                    "tps_loss", epoch_mean_tps, epoch)
                tbx_writer.add_scalar(
                    "fpc_loss", epoch_mean_fpc, epoch)
                tbx_writer.add_scalar(
                    "tv_loss", epoch_mean_tv, epoch)
                # s3fd_adv_background_image = imgconv.image_clamp(s3fd_adv_background_image, min=s3fd_util.S3FD_IMAGE_MIN, max=s3fd_util.S3FD_IMAGE_MAX)
                """
                if not (c_s3fd_adv_background_image == s3fd_adv_background_image).all():
                    print(
                        c_s3fd_adv_background_image[c_s3fd_adv_background_image != s3fd_adv_background_image])
                """
                pil_adv_background_image = transforms.functional.to_tensor(
                    s3fd_util.image_decode(s3fd_adv_background_image))
                tbx_writer.add_image(
                    "adversarial_background_image", pil_adv_background_image, epoch)

                if adv_detections_list[0] is not None:
                    pil_adv_image = s3fd_util.image_decode(
                        s3fd_adv_image_list[0], scale_list[0])
                    anno_adv_image = transforms.functional.to_tensor(imgdraw.draw_boxes(
                        pil_adv_image, adv_detections_list[0].get_image_xyxy()))
                    tbx_writer.add_image(
                        "adversarial_image", anno_adv_image, epoch)
                else:
                    anno_adv_image = transforms.functional.to_tensor(s3fd_util.image_decode(
                        s3fd_adv_image_list[0], scale_list[0]))
                    tbx_writer.add_image(
                        "adversarial_image", anno_adv_image, epoch)
    return s3fd_adv_background_image.clone().cpu()


@ hydra.main(version_base=None, config_path="../conf/", config_name="train_background")
def main(cfg: DictConfig):
    config = cfg.train_main
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    orig_wd_path = os.getcwd()

    """
    setting_path = os.path.join(orig_wd_path, config.model.setting_path)
    annfile_path = os.path.join(orig_wd_path, config.model.weight_path)
    model = yolo.load_model(
        setting_path,
        annfile_path
    )
    yolo_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((416, 416)),
    ])
    """

    weight_path = os.path.join(orig_wd_path, config.model.weight_path)
    model = s3fd_util.load_model(weight_path)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    mode = config.mode

    match mode:
        case "monitor":
            mode_config = config.monitor_mode

            image_set_path = os.path.join(
                orig_wd_path, mode_config.dataset.data_path)
            mask_image_set_path = os.path.join(
                orig_wd_path, mode_config.dataset.mask_data_path)
            image_set = DirectoryImageWithMaskDataset(
                image_set_path, mask_image_set_path, transform=transform)
            #    image_set_path, mask_image_set_path, transform=yolo_transforms)
            image_loader = torch.utils.data.DataLoader(
                image_set)

            gt_conf_list, gt_box_list = boxio.generate_integrated_xyxy_list(
                mode_config.dataset.ground_truthes_detection_path)
            ground_truthes = s3fd_util.detections_base(
                gt_conf_list.to(device=device), gt_box_list.to(device=device), is_xywh=False)

            tbx_writer = SummaryWriter(config.output_dir)

            with torch.autograd.detect_anomaly():
                adv_background_image = train_adversarial_image(
                    model, image_loader, ground_truthes, config.train_adversarial_image, tbx_writer=tbx_writer)

            tbx_writer.close()

            output_adv_path = os.path.join(
                config.output_dir, "adv_background_image.png")

            pil_image = s3fd_util.image_decode(adv_background_image)
            pil_image.save(output_adv_path)
            print("finished!")

        case "evaluate":
            mode_config = config.evaluate_mode

            image_set_path = os.path.join(
                orig_wd_path, mode_config.dataset.data_path)
            mask_image_set_path = os.path.join(
                orig_wd_path, mode_config.dataset.mask_data_path)
            image_set = DirectoryImageWithMaskDataset(
                image_set_path, mask_image_set_path, max_iter=3000, transform=transform)
            #    image_set_path, mask_image_set_path, transform=yolo_transforms)
            image_loader = torch.utils.data.DataLoader(
                image_set)

            gt_conf_list, gt_box_list = boxio.generate_integrated_xyxy_list(
                mode_config.dataset.ground_truthes_detection_path, max_iter=3000)
            ground_truthes = s3fd_util.detections_base(
                gt_conf_list.to(device=device), gt_box_list.to(device=device), is_xywh=False)

            tbx_writer = SummaryWriter(config.output_dir)

            adv_background_image = train_adversarial_image(
                model, image_loader, ground_truthes, config.train_adversarial_image, tbx_writer=tbx_writer)

            tbx_writer.close()

            output_adv_path = os.path.join(
                config.output_dir, "adv_background_image.png")

            pil_image = s3fd_util.image_decode(adv_background_image)
            pil_image.save(output_adv_path)
            print("finished!")

        case _:
            raise Exception('modeが想定されていない値です')


if __name__ == "__main__":
    main()
