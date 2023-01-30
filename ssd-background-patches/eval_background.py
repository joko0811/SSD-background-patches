import os
import shutil

import hydra
from omegaconf import DictConfig

import torch
from torchvision import transforms
import numpy as np

from tqdm import tqdm
from tensorboardX import SummaryWriter

from PIL import Image

from model import yolo, yolo_util, s3fd_util
from dataset.simple import DirectoryImageDataset
from dataset.mask import DirectoryImageWithMaskDataset

from box import boxio
from util import bgutil, evalutil
from dataset import coco
from imageutil import imgdraw, imgconv, imgseg
from evaluation.detection import calc_det_FP, calc_det_TP, data_utility_quority, ap


def save_detection(adv_background_image, model, image_loader, config: DictConfig):
    model.eval()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    gpu_adv_bg_image = adv_background_image.to(device)
    adv_bg_image_path = config.adv_bg_image_path
    save_path = os.path.dirname(adv_bg_image_path)

    # default is "ground_truth_detection"
    gt_det_save_dir = config.gt_det_save_dir
    gt_det_save_path = os.path.join(save_path, gt_det_save_dir)
    if os.path.isdir(gt_det_save_path):
        shutil.rmtree(gt_det_save_path)
    os.mkdir(gt_det_save_path)

    # default is "adversarial_background_detection"
    adv_bg_det_save_dir = config.adv_bg_det_save_dir
    adv_bg_det_save_path = os.path.join(save_path, adv_bg_det_save_dir)
    if os.path.isdir(adv_bg_det_save_path):
        shutil.rmtree(adv_bg_det_save_path)
    os.mkdir(adv_bg_det_save_path)

    for image_list, image_path_list in tqdm(image_loader):

        image_hw = image_list.shape[-2:]

        gpu_image_list = image_list.to(device)
        adv_bg_image_list = bgutil.background_applyer(
            gpu_image_list, gpu_adv_bg_image)

        gt_det_path_list = evalutil.gen_detection_path(
            image_path_list, os.path.join(save_path, gt_det_save_dir))
        evalutil.save_detection_text(
            gpu_image_list, gt_det_path_list, model, boxio.format_yolo, optional=image_hw)

        adv_bg_det_path_list = evalutil.gen_detection_path(
            image_path_list, os.path.join(save_path, adv_bg_det_save_dir))
        evalutil.save_detection_text(
            adv_bg_image_list, adv_bg_det_path_list, model, boxio.format_yolo, optional=image_hw)


def tbx_monitor(adv_background_image, model, image_loader, config):

    model.eval()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # gpu_adv_bg_image = adv_background_image.to(device)
    gpu_adv_bg_image = s3fd_util.image_encode(
        adv_background_image)[0].to(device)

    # class_names = coco.load_class_names(config.class_names_path)
    tbx_writer = SummaryWriter(config.output_dir)

    for batch_iter, (image_list, mask_image_list, _, _) in enumerate(tqdm(image_loader, leave=False)):

        # gpu_image_list = image_list.to(device)
        # adv_image_list = bgutil.background_applyer(gpu_image_list, gpu_adv_bg_image)
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

        adv_image_list = imgseg.composite_image(
            s3fd_image_list, gpu_adv_bg_image, s3fd_mask_image_list)

        adv_output = model(adv_image_list)
        # adv_nms_out = yolo_util.nms(adv_output)
        # adv_detections_list = yolo_util.make_detections_list(adv_nms_out, yolo_util.detections_yolo_loss)
        adv_detections_list = s3fd_util.make_detections_list(
            adv_output, scale_list, s3fd_util.detections_s3fd, 0.6)

        for i, adv_image in enumerate(adv_image_list):
            if adv_detections_list[i] is not None:
                anno_adv_image = imgdraw.draw_boxes(
                    s3fd_util.image_decode(adv_image, scale_list[i]), adv_detections_list[i].xyxy)
                tbx_writer.add_image("adversarial_image",
                                     transforms.functional.to_tensor(anno_adv_image), batch_iter)
    return


def evaluate_background(adv_background_image, model, image_loader, config: DictConfig):

    iou_thresh = 0.5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    s3fd_adv_background_image = s3fd_util.image_encode(
        adv_background_image)[0].to(device)

    model.eval()

    gt_box_list = list()
    gt_total_det = 0

    adv_box_list = list()
    adv_conf_list = np.array([])

    adv_total_tp = 0
    adv_total_fp = 0

    for image_list, mask_image_list, image_path_list, mask_image_path_list in tqdm(image_loader, leave=False):

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

        s3fd_adv_image_list = imgseg.composite_image(
            s3fd_image_list, s3fd_adv_background_image, s3fd_mask_image_list)

        gt_output = model(s3fd_image_list)
        gt_detections_list = s3fd_util.make_detections_list(
            gt_output, scale_list, s3fd_util.detections_s3fd, config.model_thresh)

        adv_output = model(s3fd_adv_image_list)
        adv_detections_list = s3fd_util.make_detections_list(
            adv_output, scale_list, s3fd_util.detections_s3fd, config.model_thresh)

        for gt_det, adv_det in zip(gt_detections_list, adv_detections_list):
            # 画像毎

            if adv_det is not None:
                adv_box_list.append(adv_det.xyxy.detach(
                ).cpu().resolve_conj().resolve_neg())
                adv_conf_list = np.append(adv_conf_list, adv_det.conf.detach(
                ).cpu().resolve_conj().resolve_neg().numpy())
                adv_total_fp += adv_det.total_det

            if gt_det is not None:
                gt_box_list.append(gt_det.xyxy.detach(
                ).cpu().resolve_conj().resolve_neg())
                gt_total_det += gt_det.total_det
            elif adv_det is not None:
                gt_box_list.append(torch.zeros((adv_det.total_det, 4)).to(
                    device='cpu', dtype=torch.float))

            if (adv_det is not None) and (gt_det is not None):
                adv_total_tp += calc_det_TP(adv_det.xyxy,
                                            gt_det.xyxy, iou_thresh)
                adv_total_fp += calc_det_FP(adv_det.xyxy,
                                            gt_det.xyxy, iou_thresh)

    ap_score = ap(gt_box_list, adv_box_list, adv_conf_list, iou_thresh)
    duq_score = data_utility_quority(gt_total_det, adv_total_tp, adv_total_fp)
    print('AP: %f, DUQ: %f' % (ap_score, duq_score))


@ hydra.main(version_base=None, config_path="../conf/", config_name="eval_background")
def main(cfg: DictConfig):

    config = cfg.main
    orig_wd_path = os.getcwd()

    if config.adv_bg_image_path == "":
        print("please select path")
        print(
            "Usage: python ssd-background-patches/eval_background.py path={path/to/adversarial_background_image}"
        )
        return

    adv_bg_image_path = os.path.join(orig_wd_path, config.adv_bg_image_path)
    # adv_bg_image = yolo_util.get_yolo_format_image_from_file(adv_bg_image_path)
    adv_bg_image = Image.open(adv_bg_image_path)

    # model_setting_path = os.path.join(orig_wd_path, config.model.setting_path)
    model_annfile_path = os.path.join(orig_wd_path, config.model.weight_path)
    model = s3fd_util.load_model(
        model_annfile_path
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image_set_path = os.path.join(orig_wd_path, config.dataset.data_path)
    mask_image_set_path = os.path.join(
        orig_wd_path, config.dataset.mask_data_path)
    image_set = DirectoryImageWithMaskDataset(
        image_set_path, mask_image_set_path, max_iter=6000, transform=transform)
    image_loader = torch.utils.data.DataLoader(image_set)

    with torch.no_grad():
        # save_detection(adv_bg_image, model, image_loader, config.save_detection)
        # tbx_monitor(adv_bg_image, model, image_loader, config.tbx_monitor)
        evaluate_background(adv_bg_image, model, image_loader,
                            config.evaluate_background)


if __name__ == "__main__":
    main()
