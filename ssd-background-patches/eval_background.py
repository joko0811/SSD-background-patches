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

from model import s3fd_util
from model.base_util import BackgroundBaseTrainer

from box import boxio
from util import bgutil, evalutil
from dataset import coco
from imageutil import imgdraw, imgconv, imgseg
from evaluation.detection import data_utility_quority, list_iou
from sklearn.metrics import average_precision_score


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
        adv_detections_list = s3fd_util.make_detections_list(
            adv_output,  0.6)

        for i, adv_image in enumerate(adv_image_list):

            tbx_writer.add_image("original_image",
                                 transforms.functional.to_tensor(pil_image_list[i]), batch_iter)
            tbx_writer.add_image("mask_image",
                                 transforms.functional.to_tensor(pil_mask_image_list[i]), batch_iter)
            if adv_detections_list[i] is not None:
                anno_adv_image = imgdraw.draw_boxes(
                    s3fd_util.image_decode(adv_image, scale_list[i]), adv_detections_list[i].xyxy*scale_list[i])
                tbx_writer.add_image("adversarial_image",
                                     transforms.functional.to_tensor(anno_adv_image), batch_iter)

    return


def evaluate_background(adv_background_image, trainer: BackgroundBaseTrainer, config: DictConfig):

    iou_thresh = 0.5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adv_background_image = transforms.functional.resize(
        transforms.functional.pil_to_tensor(adv_background_image), trainer.get_image_size())[0].to(device)

    image_loader = trainer.get_dataloader()
    model = trainer.load_model()
    model.eval()

    gt_total_det = 0

    adv_conf_array = np.array([])
    adv_tp_binary_array = np.array([])

    adv_total_tp = 0
    adv_total_fp = 0

    for (image_list, mask_image_list), image_info in tqdm(image_loader):

        image_list = image_list.to(
            device=device, dtype=torch.float)
        mask_image_list = mask_image_list.to(
            device=device)
        scale_list = torch.cat([image_info['width'], image_info['height'],
                                image_info['width'], image_info['height']]).T.to(device=device, dtype=torch.float)

        adv_image_list = imgseg.composite_image(
            image_list, adv_background_image, mask_image_list)

        gt_output = model(image_list)
        gt_detections_list = trainer.make_detections_list(
            gt_output, config.model_thresh)

        adv_output = model(adv_image_list)
        adv_detections_list = trainer.make_detections_list(
            adv_output, config.model_thresh)

        for gt_det, adv_det in zip(gt_detections_list, adv_detections_list):
            # 画像毎
            if (gt_det is None) and (adv_det is None):
                continue

            elif (gt_det is None) and (adv_det is not None):
                adv_tp_binary_array = np.append(
                    adv_tp_binary_array, np.zeros(len(adv_det)))
                adv_conf_array = np.append(adv_conf_array, adv_det.conf.detach(
                ).cpu().resolve_conj().resolve_neg().numpy())
                adv_total_fp += len(adv_det)
                continue

            elif (gt_det is not None) and (adv_det is None):
                gt_total_det += len(gt_det)
                continue

            elif (gt_det is not None) and (adv_det is not None):
                gt_total_det += len(gt_det)

                adv_tp_binary = (list_iou(adv_det.xyxy, gt_det.xyxy)
                                 >= iou_thresh).any(dim=1).long()

                adv_total_tp += adv_tp_binary.nonzero().shape[0]
                adv_total_fp += adv_tp_binary.shape[0] - \
                    adv_tp_binary.nonzero().shape[0]

                adv_tp_binary_array = np.append(
                    adv_tp_binary_array, adv_tp_binary.detach(
                    ).cpu().resolve_conj().resolve_neg().numpy())
                adv_conf_array = np.append(adv_conf_array, adv_det.conf.detach(
                ).cpu().resolve_conj().resolve_neg().numpy())

                continue

    np.save(os.path.join(config.output_dir, "y_true.npy"), adv_tp_binary_array)
    np.save(os.path.join(config.output_dir, "y_score.npy"), adv_conf_array)

    # ap_score = ap(gt_box_list, adv_box_list, adv_conf_list, iou_thresh)
    # ap_score = average_precision_score(tp_binary_list, adv_conf_list)
    # precision, recall, thresh = precision_recall_curve(tp_binary_list, adv_conf_list)
    # disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    # disp.plot()
    # plt.show()

    duq_score = data_utility_quority(gt_total_det, adv_total_tp, adv_total_fp)
    ap_score = average_precision_score(adv_tp_binary_array, adv_conf_array)

    print("GT_FACES: "+str(gt_total_det))
    print("TP: "+str(adv_total_tp))
    print("FP: "+str(adv_total_fp))

    print("DUQ: "+str(duq_score))
    print("AP: "+str(ap_score))


@ hydra.main(version_base=None, config_path="../conf/", config_name="eval_background")
def main(cfg: DictConfig):

    config = cfg.main
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainer: BackgroundBaseTrainer = hydra.utils.instantiate(
        config.trainer)

    adv_bg_image_path = config.adv_bg_image_path
    adv_bg_image = Image.open(adv_bg_image_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    with torch.no_grad():
        # save_detection(adv_bg_image, model, image_loader, config.save_detection)
        # tbx_monitor(adv_bg_image, model, image_loader, config.tbx_monitor)
        evaluate_background(adv_bg_image, trainer,
                            config.evaluate_background)


if __name__ == "__main__":
    main()
