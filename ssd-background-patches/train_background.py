import os

import hydra
from omegaconf import DictConfig

import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets.coco import CocoDetection

from tensorboardX import SummaryWriter
from tqdm import tqdm

from model import yolo, yolo_util
from dataset.simple import DirectoryImageDataset
from loss.dt_based_loss import total_loss
from imageutil import imgseg, imgconv
from dataset import coco
from box import boxio


def train_adversarial_image(model, image_loader, config: DictConfig, class_names=None, tbx_writer=None):

    max_epoch = config.max_epoch  # default 250

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 敵対的背景
    adv_background_image = torch.zeros((3, 416, 416), device=device)

    # TODO:最適化は使っていないので置き換える(optimizer.zero_gradは使っているので注意)
    optimizer = optim.Adam([adv_background_image])

    for epoch in tqdm(range(max_epoch)):
        for image_list, _ in tqdm(image_loader):

            adv_background_image.requires_grad = True

            # Preprocessing
            # Set to no_grad since the process is not needed for gradient calculation.
            with torch.no_grad():
                gpu_image_list = image_list.to(
                    device=device, dtype=torch.float)
                mask_image_list = imgseg.gen_mask_image(gpu_image_list)

            adv_image_list = imgseg.composite_image(
                gpu_image_list, adv_background_image, mask_image_list)

            # Set to no_grad since the process is not needed for gradient calculation.
            with torch.no_grad():
                # Detection from unprocessed images
                # Detection of unprocessed images as Groundtruth
                gt_output = model(gpu_image_list)
                gt_nms_out = yolo_util.nms(gt_output)
                gt_detections_list = yolo_util.detections_loss(gt_nms_out[0])

            # Detection from adversarial images
            adv_output = model(adv_image_list)
            adv_nms_out = yolo_util.nms(adv_output)
            adv_detections_list = yolo_util.detections_loss(adv_nms_out[0])

            tpc_loss, tps_loss, fpc_loss, end_flag = total_loss()
            loss = tpc_loss+tps_loss+fpc_loss

            if end_flag:
                # 損失計算時に条件を満たした場合終了(zの要素がすべて0の場合)
                # 原因は特定していないがzの要素が全て0の場合勾配画像がバグるので
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
                    "tpc_loss", tpc_loss, epoch)
                tbx_writer.add_scalar(
                    "tps_loss", tps_loss, epoch)
                tbx_writer.add_scalar(
                    "fpc_loss", fpc_loss, epoch)

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
            image_set_path = os.path.join(
                orig_wd_path, config.dataset.data_path)
            image_set = DirectoryImageDataset(
                image_set_path, transform=yolo_transforms)
            image_loader = torch.utils.data.DataLoader(image_set, batch_size=2)

            class_names_path = os.path.join(
                orig_wd_path, config.dataset.class_names)
            class_names = coco.load_class_names(class_names_path)
            tbx_writer = SummaryWriter(config.output_dir)

            with torch.autograd.detect_anomaly():
                adv_image, _, _ = train_adversarial_image(
                    model, image_loader, config.train_adversarial_image, class_names=class_names, tbx_writer=tbx_writer)

            tbx_writer.close()

            output_adv_path = os.path.join(
                config.output_dir, "adv_image.png")

            pil_image = transforms.functional.to_pil_image(adv_image[0])
            pil_image.save(output_adv_path)
            print("finished!")

        case "evaluate":
            iterate_num = 2000
            iterate_digit = len(str(iterate_num))

            os.mkdir(config.evaluate_orig_path)
            os.mkdir(config.evaluate_image_path)
            os.mkdir(config.evaluate_detection_path)
            os.mkdir(config.evaluate_patch_path)

            coco_path = os.path.join(orig_wd_path, config.dataset.data_path)
            coco_annfile_path = os.path.join(
                orig_wd_path, config.dataset.annfile_path)

            train_set = CocoDetection(root=coco_path,
                                      annFile=coco_annfile_path, transform=yolo_transforms)
            train_loader = torch.utils.data.DataLoader(train_set)

            for image_idx, (image, _) in tqdm(enumerate(train_loader), total=iterate_num):
                if image_idx >= iterate_num:
                    break
                adv_image, detections, background_patch_boxes = train_adversarial_image(
                    model, image, config.train_adversarial_image)

                # 結果の保存
                iter_str = str(image_idx).zfill(iterate_digit)

                output_orig_path = os.path.join(
                    config.evaluate_orig_path, f'{iter_str}.png')
                output_adv_path = os.path.join(
                    config.evaluate_image_path, f'{iter_str}.png')
                output_detections_path = os.path.join(
                    config.evaluate_detections_path, f'{iter_str}.txt')
                output_patch_path = os.path.join(
                    config.evaluate_patch_path, f'{iter_str}.txt')

                orig_pil_image = transforms.functional.to_pil_image(image[0])
                orig_pil_image.save(output_orig_path)
                adv_pil_image = transforms.functional.to_pil_image(
                    adv_image[0])
                adv_pil_image.save(output_adv_path)
                detstr = boxio.format_boxes(detections.xyxy)
                with open(output_detections_path, 'w') as f:
                    f.write(detstr)
                bpstr = boxio.format_boxes(background_patch_boxes)
                with open(output_patch_path, 'w') as f:
                    f.write(bpstr)

        case _:
            raise Exception('modeが想定されていない値です')


if __name__ == "__main__":
    main()
