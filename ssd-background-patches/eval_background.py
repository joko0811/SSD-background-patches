import os
import shutil
import random

import hydra
from omegaconf import DictConfig, open_dict

import torch
from torchvision import transforms
import numpy as np
import pandas as pd

from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn.metrics import average_precision_score


from box.boxio import format_detections
from imageutil import imgdraw
from model.base_util import BackgroundBaseTrainer
from ptmanager.base_patch import BaseBackgroundManager
from detection.detection_base import DetectionsBase
from detection.tp_fp_manager import TpFpManager
from evaluation.detection import data_utility_quority, f1, precision, recall, list_iou
from box.boxconv import xyxy2xywh


def save_detection(
    adv_patch,
    background_manager: BaseBackgroundManager,
    trainer: BackgroundBaseTrainer,
    config: DictConfig,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_loader = trainer.get_dataloader()
    model = trainer.load_model()
    model.eval()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    adv_patch = adv_patch.to(device)

    patch_size = adv_patch.shape[1:]

    det_output_path = os.path.join(config.output_dir, "detections")
    if not os.path.exists(det_output_path):
        os.mkdir(det_output_path)

    for (image_list, mask_image_list), image_info in tqdm(image_loader):
        image_list = image_list.to(device=device, dtype=torch.float)
        mask_image_list = mask_image_list.to(device=device)

        image_size = image_list[0].shape[1:]  # (H,W)
        args_of_tpatch = background_manager.generate_kwargs_of_transform_patch(
            image_size, patch_size, xyxy2xywh(image_info["xyxy"])[:, :, 2:]
        )
        (
            tmp_adv_background_image,
            adv_background_mask,
        ) = background_manager.transform_patch(
            adv_patch / 255, image_size, **args_of_tpatch
        )
        adv_background_image = tmp_adv_background_image * 255

        adv_image_list = background_manager.apply(
            adv_background_image, adv_background_mask, image_list, mask_image_list
        ).to(dtype=torch.float)

        adv_output = model(adv_image_list)
        adv_detections_list = trainer.make_detections_list(
            adv_output, config.model_thresh
        )

        for image_idx, det in enumerate(adv_detections_list):
            if det is None:
                continue

            det_str = format_detections(det)

            # ファイルまでのパス、拡張子を除いたファイル名を取得
            image_name = os.path.basename(image_info["path"][image_idx]).split(".")[0]
            det_file_path = os.path.join(det_output_path, image_name + ".txt")

            with open(det_file_path, "w") as f:
                f.write(det_str)


def tbx_monitor(
    adv_patch,
    background_manager: BaseBackgroundManager,
    trainer: BackgroundBaseTrainer,
    config: DictConfig,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_loader = trainer.get_dataloader()
    model = trainer.load_model()
    model.eval()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    adv_patch = adv_patch.to(device)

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

        image_size = image_list[0].shape[1:]  # (H,W)

        patch_size = adv_patch.shape[1:]

        # パッチ変形用のy引数用意
        # パッチのランダムな座標変動
        args_of_tpatch = background_manager.generate_kwargs_of_transform_patch(
            image_size, patch_size, xyxy2xywh(image_info["xyxy"])[:, :, 2:]
        )
        (
            tmp_adv_background_image,
            adv_background_mask,
        ) = background_manager.transform_patch(
            adv_patch / 255, image_size, **args_of_tpatch
        )
        adv_background_image = tmp_adv_background_image * 255

        adv_image_list = background_manager.apply(
            adv_background_image, adv_background_mask, image_list, mask_image_list
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

            if adv_detections_list[i] is not None and gt_det is not None:
                det_gt_iou = list_iou(adv_detections_list[i].xyxy, gt_det.xyxy)

                tp_det = adv_detections_list[i].xyxy[(det_gt_iou >= 0.6).any(dim=1)]
                anno_adv_image = imgdraw.draw_boxes(
                    anno_adv_image,
                    # adv_detections_list[i].xyxy * scale_list[i],
                    tp_det * scale_list[i],
                    color=(25, 139, 95),  # green
                )
                fp_det = adv_detections_list[i].xyxy[(det_gt_iou < 0.5).all(dim=1)]
                anno_adv_image = imgdraw.draw_boxes(
                    anno_adv_image,
                    # adv_detections_list[i].xyxy * scale_list[i],
                    fp_det * scale_list[i],
                    color=(255, 0, 0),  # red
                )
            elif adv_detections_list[i] is not None and gt_det is None:
                fp_det = adv_detections_list[i].xyxy
                anno_adv_image = imgdraw.draw_boxes(
                    anno_adv_image,
                    # adv_detections_list[i].xyxy * scale_list[i],
                    fp_det * scale_list[i],
                    color=(255, 0, 0),
                )

            if gt_det is not None:
                anno_adv_image = imgdraw.draw_boxes(
                    anno_adv_image,
                    gt_det.xyxy * scale_list[i],
                    score=gt_det.conf,
                    color=(255, 255, 255),
                )

            tbx_writer.add_image(
                "adversarial_image",
                transforms.functional.to_tensor(anno_adv_image),
                batch_iter,
            )


def evaluate_background(
    adv_patch,
    background_manager: BaseBackgroundManager,
    trainer: BackgroundBaseTrainer,
    config: DictConfig,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adv_patch = adv_patch.to(device)

    image_loader = trainer.get_dataloader()
    model = trainer.load_model()
    model.eval()

    tp_fp_manager = TpFpManager(device=device)

    transform = transforms.Compose(
        [transforms.ColorJitter(brightness=0.1, saturation=0.1)]
    )

    columns = ["image_path", "tp_conf", "gt", "tp", "fp", "fn"]
    eval_result = pd.DataFrame(columns=columns).set_index("image_path")

    for (image_list, mask_image_list), image_info in tqdm(image_loader):
        image_list = image_list.to(device=device, dtype=torch.float)
        mask_image_list = mask_image_list.to(device=device)

        image_size = image_list[0].shape[1:]  # (H,W)

        patch_size = adv_patch.shape[1:]
        args_of_tpatch = background_manager.generate_kwargs_of_transform_patch(
            image_size, patch_size, xyxy2xywh(image_info["xyxy"])[:, :, 2:]
        )
        (
            tmp_adv_background_image,
            adv_background_mask,
        ) = background_manager.transform_patch(
            adv_patch / 255, image_size, **args_of_tpatch
        )
        adv_background_image = tmp_adv_background_image * 255

        adv_image_list = background_manager.apply(
            adv_background_image, adv_background_mask, image_list, mask_image_list
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
            tp_conf, tp, fp, fn, gt = tp_fp_manager.add_detection(adv_det, gt_det)
            row = pd.DataFrame(
                [[os.path.abspath(image_info["path"][0]), tp_conf, gt, tp, fp, fn]],
                columns=columns,
            ).set_index("image_path")
            eval_result = pd.concat([eval_result, row])

    eval_result.to_csv(os.path.join(config.output_dir, "eval_result.csv"))

    tp, fp, fn, gt = tp_fp_manager.get_value()
    duq_score = data_utility_quority(gt, tp, fp)

    adv_tp_binary_array, adv_conf_array = tp_fp_manager.get_sklearn_y_true_score()
    if adv_tp_binary_array.size != 0 and adv_conf_array.size != 0:
        ap_score = average_precision_score(adv_tp_binary_array, adv_conf_array)

        np.save(os.path.join(config.output_dir, "y_true.npy"), adv_tp_binary_array)
        np.save(os.path.join(config.output_dir, "y_score.npy"), adv_conf_array)
    else:
        print("tp is not found")
        ap_score = 0

    precision_score = precision(tp, fp)
    recall_score = recall(tp, fn)

    f_score = f1(precision_score, recall_score)

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
        + "F: "
        + str(f_score)
        + "\n"
    )

    print(result)
    with open(os.path.join(config.output_dir, "result.txt"), mode="w") as f:
        f.write(result)


@hydra.main(version_base=None, config_path="../conf/", config_name="eval_background")
def main(cfg: DictConfig):
    trainer: BackgroundBaseTrainer = hydra.utils.instantiate(cfg.trainer)

    adv_bg_image_path = cfg.adv_bg_image_path
    adv_patch = torch.load(adv_bg_image_path)

    with open_dict(cfg):
        cfg.ptmanager._partial_ = True

    background_manager: BaseBackgroundManager = hydra.utils.instantiate(cfg.ptmanager)(
        patch_size=adv_patch.shape[1:]
    )

    patch_postprocesser = (
        hydra.utils.call(cfg.pttransforms) if "_target_" in cfg.pttransforms else None
    )

    # default is None
    background_manager.set_patch_postprosesser(patch_postprosesser=patch_postprocesser)

    with torch.no_grad():
        # save_detection(adv_patch, background_manager, trainer, cfg.evaluate_background)
        # tbx_monitor(adv_patch, background_manager, trainer, cfg.evaluate_background)
        evaluate_background(
            adv_patch,
            background_manager,
            trainer,
            cfg.evaluate_background,
        )


if __name__ == "__main__":
    main()
