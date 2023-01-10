import os

import hydra
from omegaconf import DictConfig

import torch
import torch.optim as optim
from torchvision import transforms

from tensorboardX import SummaryWriter
from tqdm import tqdm

from model import yolo, yolo_util
from dataset.mask import DirectoryImageWithMaskDataset
from loss import proposed
from imageutil import imgseg, imgconv, imgdraw
from dataset import coco


def train_adversarial_image(model, image_loader, config: DictConfig, class_names=None, tbx_writer=None):

    max_epoch = config.max_epoch  # default 250

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 敵対的背景
    adv_background_image = torch.zeros((3, 416, 416), device=device)

    # TODO:最適化は使っていないので置き換える(optimizer.zero_gradは使っているので注意)
    optimizer = optim.Adam([adv_background_image])

    for epoch in tqdm(range(max_epoch)):
        for image_list, mask_image_list, _, _ in tqdm(image_loader, leave=False):

            # Preprocessing
            # Set to no_grad since the process is not needed for gradient calculation.
            with torch.no_grad():
                gpu_image_list = image_list.to(
                    device=device, dtype=torch.float)
                gpu_mask_image_list = mask_image_list.to(
                    device=device, dtype=torch.float)

                # Detection from unprocessed images
                # Detection of unprocessed images as Groundtruth
                gt_output = model(gpu_image_list)
                gt_nms_out = yolo_util.nms(gt_output)
                gt_detections_list = yolo_util.make_detections_list(
                    gt_nms_out, yolo_util.detections_base)

            adv_background_image.requires_grad = True
            # adv_image_list = bgutil.background_applyer(gpu_image_list, adv_background_image)
            adv_image_list = imgseg.composite_image(
                gpu_image_list, adv_background_image, gpu_mask_image_list)

            # Detection from adversarial images
            adv_output = model(adv_image_list)
            adv_nms_out = yolo_util.nms(adv_output)
            adv_detections_list = yolo_util.make_detections_list(
                adv_nms_out, yolo_util.detections_loss)

            tpc_loss_list = torch.zeros(
                image_loader.batch_size, device=device)
            tps_loss_list = torch.zeros(
                image_loader.batch_size, device=device)
            fpc_loss_list = torch.zeros(
                image_loader.batch_size, device=device)

            for i in range(image_loader.batch_size):
                if adv_detections_list[i] is None:
                    tpc_loss_list[i] += 0
                    tps_loss_list[i] += 0
                    fpc_loss_list[i] += 0
                    continue

                tpc_loss, tps_loss, fpc_loss, end_flag = proposed.total_loss(
                    adv_detections_list[i], gt_detections_list[i], (416, 416), config.loss)

                if end_flag:
                    tpc_loss_list[i] += 0
                    tps_loss_list[i] += 0
                    fpc_loss_list[i] += 0
                else:
                    tpc_loss_list[i] += tpc_loss
                    tps_loss_list[i] += tps_loss
                    fpc_loss_list[i] += fpc_loss

            max_tpc = torch.max(tpc_loss_list)
            max_tps = torch.max(tps_loss_list)
            max_fpc = torch.max(fpc_loss_list)

            loss = max_tpc+max_tps+max_fpc

            if loss == 0:
                break

            optimizer.zero_grad()
            loss.backward()
            # The Adversarial background image is updated here
            optimizer.step()

        with torch.no_grad():
            if tbx_writer is not None:
                tbx_writer.add_scalar(
                    "total_loss", loss, epoch)
                tbx_writer.add_scalar(
                    "tpc_loss", max_tpc, epoch)
                tbx_writer.add_scalar(
                    "tps_loss", max_tps, epoch)
                tbx_writer.add_scalar(
                    "fpc_loss", max_fpc, epoch)
                if (epoch % 10 == 0):
                    for i, (adv_image, adv_detections) in enumerate(zip(adv_image_list, adv_detections_list)):
                        if adv_detections is not None:
                            anno_adv_image = imgdraw.draw_annotations(
                                adv_image, adv_detections, class_names)
                            tbx_writer.add_image(
                                "adversarial_image_"+str(i), anno_adv_image, epoch)
    return adv_background_image.clone().cpu()


@ hydra.main(version_base=None, config_path="../conf/", config_name="train_background")
def main(cfg: DictConfig):
    config = cfg.train_main

    orig_wd_path = os.getcwd()

    setting_path = os.path.join(orig_wd_path, config.model.setting_path)
    annfile_path = os.path.join(orig_wd_path, config.model.weight_path)
    model = yolo.load_model(
        setting_path,
        annfile_path)
    model.eval()

    yolo_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((416, 416)),
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
                image_set_path, mask_image_set_path, transform=yolo_transforms)
            image_loader = torch.utils.data.DataLoader(
                image_set, batch_size=6)

            class_names_path = os.path.join(
                orig_wd_path, mode_config.dataset.class_names)
            class_names = coco.load_class_names(class_names_path)
            tbx_writer = SummaryWriter(config.output_dir)

            with torch.autograd.detect_anomaly():
                adv_background_image = train_adversarial_image(
                    model, image_loader, config.train_adversarial_image, class_names=class_names, tbx_writer=tbx_writer)

            tbx_writer.close()

            output_adv_path = os.path.join(
                config.output_dir, "adv_background_image.png")

            pil_image = imgconv.tensor2pil(adv_background_image)
            pil_image.save(output_adv_path)
            print("finished!")

        case "evaluate":
            mode_config = config.evaluate_mode

            image_set_path = os.path.join(
                orig_wd_path, mode_config.dataset.data_path)
            mask_image_set_path = os.path.join(
                orig_wd_path, mode_config.dataset.mask_data_path)
            image_set = DirectoryImageWithMaskDataset(
                image_set_path, mask_image_set_path, transform=yolo_transforms)
            image_loader = torch.utils.data.DataLoader(image_set, batch_size=2)

            adv_background_image = train_adversarial_image(
                model, image_loader, config.train_adversarial_image)

            output_adv_path = os.path.join(
                config.output_dir, "adv_background_image.png")

            pil_image = imgconv.tensor2pil(adv_background_image)
            pil_image.save(output_adv_path)
            print("finished!")

        case _:
            raise Exception('modeが想定されていない値です')


if __name__ == "__main__":
    main()
