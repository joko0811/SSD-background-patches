import os
import shutil

import hydra
from omegaconf import DictConfig

import torch

from tqdm import tqdm

from model import yolo, yolo_util
from dataset.simple import DirectoryImageDataset

from box import boxio
from util import bgutil, evalutil


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
    adv_bg_image = yolo_util.get_yolo_format_image_from_file(adv_bg_image_path)

    model_setting_path = os.path.join(orig_wd_path, config.model.setting_path)
    model_annfile_path = os.path.join(orig_wd_path, config.model.weight_path)
    model = yolo.load_model(
        model_setting_path,
        model_annfile_path
    )

    image_set_path = os.path.join(orig_wd_path, config.dataset.data_path)
    image_set = DirectoryImageDataset(
        image_set_path, transform=yolo_util.YOLO_TRANSFORMS)
    image_loader = torch.utils.data.DataLoader(image_set)

    save_detection(adv_bg_image, model, image_loader,
                   config.save_detection)


if __name__ == "__main__":
    main()
