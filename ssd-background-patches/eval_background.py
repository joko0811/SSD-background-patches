import os
import shutil

import hydra
from omegaconf import DictConfig

import torch
from torchvision import transforms
import numpy as np

from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn.metrics import average_precision_score


from box import boxio
from util import bgutil, evalutil
from imageutil import imgdraw
from model.base_util import BackgroundBaseTrainer
from patch_manager import BaseBackgroundManager
from detection.detection_base import DetectionsBase
from detection.tp_fp_manager import TpFpManager
from evaluation.detection import data_utility_quority, f1, precision, recall, list_iou


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
        adv_bg_image_list = bgutil.background_applyer(gpu_image_list, gpu_adv_bg_image)

        gt_det_path_list = evalutil.gen_detection_path(
            image_path_list, os.path.join(save_path, gt_det_save_dir)
        )
        evalutil.save_detection_text(
            gpu_image_list,
            gt_det_path_list,
            model,
            boxio.format_yolo,
            optional=image_hw,
        )

        adv_bg_det_path_list = evalutil.gen_detection_path(
            image_path_list, os.path.join(save_path, adv_bg_det_save_dir)
        )
        evalutil.save_detection_text(
            adv_bg_image_list,
            adv_bg_det_path_list,
            model,
            boxio.format_yolo,
            optional=image_hw,
        )


def tbx_monitor(
    adv_background,
    background_manager: BaseBackgroundManager,
    trainer: BackgroundBaseTrainer,
    config: DictConfig,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_loader = trainer.get_dataloader()
    model = trainer.load_model()
    model.eval()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    adv_background = adv_background.to(device)

    tbx_writer = SummaryWriter(config.output_dir)

    for batch_iter, ((image_list, mask_image_list), image_info) in enumerate(
        tqdm(image_loader)
    ):
        scale_list = torch.cat(
            [
                image_info["width"],
                image_info["height"],
                image_info["width"],
                image_info["height"],
            ]
        ).T.to(device=device, dtype=torch.float)

        image_list = image_list.to(device=device, dtype=torch.float)
        mask_image_list = mask_image_list.to(device=device)

        adv_image_list = background_manager.apply(
            adv_background, image_list, mask_image_list
        ).to(dtype=torch.float)

        adv_output = model(adv_image_list)
        adv_detections_list = trainer.make_detections_list(
            adv_output, config.model_thresh
        )

        for i, adv_image in enumerate(adv_image_list):
            if (image_info["conf"][i].nelement() != 0) and (
                image_info["xyxy"][i].nelement() != 0
            ):
                gt_det = DetectionsBase(
                    image_info["conf"][i], image_info["xyxy"][i], is_xywh=False
                )
            else:
                gt_det = None

            anno_adv_image = trainer.transformed2pil(
                adv_image, (image_info["height"][i], image_info["width"][i])
            )

            if adv_detections_list is not None:
                det_gt_iou = list_iou(adv_detections_list[i].xyxy, gt_det.xyxy)

                tp_det = adv_detections_list[i].xyxy[(det_gt_iou >= 0.5).any(dim=1)]
                anno_adv_image = imgdraw.draw_boxes(
                    anno_adv_image,
                    # adv_detections_list[i].xyxy * scale_list[i],
                    tp_det * scale_list[i],
                    color=(255, 0, 0),
                )

            if gt_det is not None:
                anno_adv_image = imgdraw.draw_boxes(
                    anno_adv_image, gt_det.xyxy * scale_list[i], color=(25, 139, 95)
                )

            tbx_writer.add_image(
                "adversarial_image",
                transforms.functional.to_tensor(anno_adv_image),
                batch_iter,
            )


def evaluate_background(
    adv_background,
    background_manager: BaseBackgroundManager,
    trainer: BackgroundBaseTrainer,
    config: DictConfig,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adv_background = adv_background.to(device)

    image_loader = trainer.get_dataloader()
    model = trainer.load_model()
    model.eval()

    tp_fp_manager = TpFpManager()

    for (image_list, mask_image_list), image_info in tqdm(image_loader):
        image_list = image_list.to(device=device, dtype=torch.float)
        mask_image_list = mask_image_list.to(device=device)

        adv_image_list = background_manager.apply(
            adv_background, image_list, mask_image_list
        )

        adv_output = model(adv_image_list)
        adv_detections_list = trainer.make_detections_list(
            adv_output, config.model_thresh
        )

        for i, adv_det in enumerate(adv_detections_list):
            # 画像毎
            if (image_info["conf"][i].nelement() != 0) and (
                image_info["xyxy"][i].nelement() != 0
            ):
                gt_det = DetectionsBase(
                    image_info["conf"][i], image_info["xyxy"][i], is_xywh=False
                )
            else:
                gt_det = None
            tp_fp_manager.add_detection(adv_det, gt_det)

    tp, fp, fn, gt = tp_fp_manager.get_value()
    duq_score = data_utility_quority(gt, tp, fp)

    adv_tp_binary_array, adv_conf_array = tp_fp_manager.get_sklearn_y_true_score()
    ap_score = average_precision_score(adv_tp_binary_array, adv_conf_array)

    np.save(os.path.join(config.output_dir, "y_true.npy"), adv_tp_binary_array)
    np.save(os.path.join(config.output_dir, "y_score.npy"), adv_conf_array)

    precision_score = precision(tp, fp)
    recall_score = recall(tp, fn)

    beta_list = np.array([0.001, 0.01, 0.1, 0.5])
    fbeta_list = np.array([])
    for beta in beta_list:
        fbeta_score = f1(precision_score, recall_score, beta)
        fbeta_list = np.append(fbeta_list, fbeta_score)

    # ap_score = ap(gt_box_list, adv_box_list, adv_conf_list, iou_thresh)
    # ap_score = average_precision_score(tp_binary_list, adv_conf_list)
    # precision, recall, thresh = precision_recall_curve(tp_binary_list, adv_conf_list)
    # disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    # disp.plot()
    # plt.show()

    result = (
        "GT_FACES: "
        + str(gt)
        + "\n"
        + "TP: "
        + str(tp)
        + "\n"
        + "FP: "
        + str(fp)
        + "\n"
        + "FN: "
        + str(fn)
        + "\n"
        + "DUQ: "
        + str(duq_score)
        + "\n"
        + "AP: "
        + str(ap_score)
        + "\n"
        + "beta: "
        + np.array2string(beta_list)
        + "\n"
        + "AF_{\\beta}"
        + np.array2string(fbeta_list)
        + "\n"
    )

    print(result)
    with open(os.path.join(config.output_dir, "result.txt"), mode="w") as f:
        f.write(result)


@hydra.main(version_base=None, config_path="../conf/", config_name="eval_background")
def main(cfg: DictConfig):
    trainer: BackgroundBaseTrainer = hydra.utils.instantiate(cfg.trainer)
    background_manager: BaseBackgroundManager = hydra.utils.instantiate(
        cfg.patch_manager
    )(trainer.get_image_size(), mode="test")

    adv_bg_image_path = cfg.adv_bg_image_path
    adv_background = background_manager.transform_patch(torch.load(adv_bg_image_path))

    with torch.no_grad():
        # save_detection(adv_bg_image, model, image_loader, config.save_detection)
        # tbx_monitor(
        #     adv_background, background_manager, trainer, cfg.evaluate_background
        # )
        evaluate_background(
            adv_background, background_manager, trainer, cfg.evaluate_background
        )


if __name__ == "__main__":
    main()
