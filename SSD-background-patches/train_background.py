import os

from PIL import Image
import numpy as np
import hydra
from omegaconf import DictConfig

import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets.coco import CocoDetection

from skimage.metrics import peak_signal_noise_ratio

from tensorboardX import SummaryWriter
from tqdm import tqdm

import proposed_func as pf
from model import yolo, yolo_util
from loss.dt_based_loss import total_loss
from util import clustering
from imageutil import imgdraw, imgseg
from dataset import coco
from box import boxio


def get_image_from_file(image_path):
    pil_image = Image.open(image_path)
    yolo_transforms = transforms.Compose([
        transforms.Resize((416, 416)),
    ])
    tensor_image = yolo_transforms(pil_image)
    return tensor_image


def train_adversarial_image(model, orig_img, config: DictConfig, class_names=None, tbx_writer=None):

    perturbate_iter = 0  # initialize
    max_perturbate_iter = config.max_iter  # default 250
    psnr_threshold = config.psnr_threshold  # default 35
    # グループ毎に割り当てられるパッチ枚数
    n_b = config.n_b  # default 3

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 元画像 更新なし
    ground_truth_image = transforms.functional.to_tensor(
        orig_img).to(device=device, dtype=torch.float)
    if ground_truth_image.dim() == 3:
        ground_truth_image = ground_truth_image.unsqueeze(0)

    # 敵対的背景 更新あり
    background_image = torch.zeros(ground_truth_image.shape, device=device)
    # マスク 更新なし PIL
    mask_image = imgseg.gen_mask_image(orig_img)
    # 前景 更新なし
    foreground_image = imgseg.wrap_composite_image(
        ground_truth_image[0], background_image[0], mask_image).unsqueeze(0)

    # 敵対的画像 更新あり
    adv_image = imgseg.wrap_composite_image(
        ground_truth_image[0], background_image[0], mask_image).unsqueeze(0)

    with torch.no_grad():
        # 素の画像を物体検出器にかけた時の出力をground truthとする
        gt_yolo_out = model(adv_image)
        gt_nms_out = yolo_util.nms(gt_yolo_out)
        if gt_nms_out[0].nelement() == 0:
            # 元画像の検出がない場合は敵対的画像を生成できない
            return background_image
        ground_truthes = yolo_util.detections_ground_truth(gt_nms_out[0])

        # ground truthesをクラスタリング
        group_labels = clustering.object_grouping(
            ground_truthes.xywh.detach().cpu().numpy())
        ground_truthes.set_group_info(torch.from_numpy(
            group_labels.astype(np.float32)).to(ground_truth_image.device))

    background_patch_boxes = torch.zeros(
        (ground_truthes.total_group*n_b, 4), device=device)

    # TODO:最適化は使っていないので置き換える(optimizer.zero_gradは使っているので注意)
    optimizer = optim.Adam([adv_image])

    for perturbate_iter in tqdm(range(max_perturbate_iter), leave=(tbx_writer is not None)):

        adv_image.requires_grad = True

        # perturbate_iter回目の敵対的画像から物体検出する
        output = model(adv_image)
        nms_out = yolo_util.nms(output)
        detections = yolo_util.detections_loss(nms_out[0])
        if nms_out[0].nelement() == 0:
            # 検出がない場合は終了
            return background_image

        tpc_loss, tps_loss, fpc_loss, end_flag = total_loss(
            detections, ground_truthes, background_patch_boxes, adv_image.shape[2:], config.loss)
        loss = tpc_loss+tps_loss+fpc_loss

        if end_flag:
            # 損失計算時に条件を満たした場合終了(zの要素がすべて0の場合)
            # 原因は特定していないがzの要素が全て0の場合勾配画像がバグるので
            break

        optimizer.zero_grad()
        loss.backward()

        gradient_image = adv_image.grad

        with torch.no_grad():

            if perturbate_iter == 0:
                # ループの最初にのみ実行
                # パッチ領域を決定する
                # NOTE:十分な背景領域が存在しない場合、パッチは選択されない
                background_patch_boxes = pf.initial_background_patches(
                    ground_truthes, gradient_image, config.initial_background_patches)
            else:
                # パッチ領域を拡大する（縮小はしない）
                background_patch_boxes = pf.expanded_background_patches(
                    background_patch_boxes, ground_truthes, gradient_image, config.expanded_background_patches)

            # 勾配画像をパッチ領域の形に切り出す
            perturbated_image = pf.perturbation_in_background_patches(
                gradient_image, background_patch_boxes)
            # make_box_image(perturbated_image, background_patch_boxes)
            # パッチの正規化
            nomalized_perturbated_image = pf.perturbation_normalization(
                perturbated_image, config.perturbation_normalization)
            # make_box_image(perturbated_image, background_patch_boxes)
            # adv_image-perturbated_imageの計算結果を[0,255]にクリップする
            background_image = pf.update_i_with_pixel_clipping(
                background_image, nomalized_perturbated_image)

            # 敵対的背景の合成
            adv_image = imgseg.wrap_composite_image(
                ground_truth_image[0], background_image[0], mask_image).unsqueeze(0)

            # psnr評価用の画像を切り出す
            psnr_truth_image = pf.perturbation_in_background_patches(
                foreground_image, background_patch_boxes).detach().cpu().numpy()
            psnr_eval_image = pf.perturbation_in_background_patches(
                adv_image, background_patch_boxes).detach().cpu().numpy()

            # NOTE:psnrはzの要素が全て0になったときにnanになることがある
            psnr = peak_signal_noise_ratio(psnr_truth_image, psnr_eval_image)

            if tbx_writer is not None:
                tbx_writer.add_scalar(
                    "total_loss", loss, perturbate_iter)
                tbx_writer.add_scalar(
                    "tpc_loss", tpc_loss, perturbate_iter)
                tbx_writer.add_scalar(
                    "tps_loss", tps_loss, perturbate_iter)
                tbx_writer.add_scalar(
                    "fpc_loss", fpc_loss, perturbate_iter)
                tbx_writer.add_scalar(
                    "psnr", psnr, perturbate_iter)

                if perturbate_iter % 10 == 0:

                    adv_anno_image = transforms.functional.to_tensor(imgdraw.tensor2annotation_image(
                        adv_image, detections, class_names))
                    tbx_writer.add_image(
                        "adversarial_image", adv_anno_image, perturbate_iter)

                    bp_image = transforms.functional.to_tensor(imgdraw.tensor2box_annotation_image(
                        nomalized_perturbated_image, background_patch_boxes))
                    tbx_writer.add_image(
                        "background_patch_boxes", bp_image, perturbate_iter)

            if psnr < psnr_threshold:
                # psnrが閾値以下
                break

    return background_image.clone().cpu(), detections, background_patch_boxes


@ hydra.main(version_base=None, config_path="../conf/", config_name="train")
def main(cfg: DictConfig):
    config = cfg.train_main

    print("change working directory"+os.getcwd())
    orig_wd_path = os.getcwd()

    setting_path = os.path.join(orig_wd_path, config.model.setting_path)
    annfile_path = os.path.join(orig_wd_path, config.model.weight_path)
    model = yolo.load_model(
        setting_path,
        annfile_path)
    model.eval()

    mode = config.mode

    match mode:
        case "monitor":

            input_image_path = os.path.join(
                orig_wd_path, config.monitor_image_path)
            image = get_image_from_file(input_image_path)

            class_names_path = os.path.join(
                orig_wd_path, config.dataset.class_names)
            class_names = coco.load_class_names(class_names_path)
            tbx_writer = SummaryWriter(config.output_dir)

            with torch.autograd.detect_anomaly():
                adv_image, _, _ = train_adversarial_image(
                    model, image, config.train_adversarial_image, class_names=class_names, tbx_writer=tbx_writer)

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

            yolo_transforms = transforms.Compose([
                transforms.Resize((416, 416)),
            ])

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
