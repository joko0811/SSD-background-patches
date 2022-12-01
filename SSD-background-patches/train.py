import os
import time
import argparse

import cv2
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets.coco import CocoDetection

from skimage.metrics import peak_signal_noise_ratio
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS

from tensorboardX import SummaryWriter
from tqdm import tqdm

import proposed_func as pf
from model import yolo, yolo_util
from loss import total_loss
from util import img, clustering
from dataset import coco


def get_image_from_file(image_path):
    # image_path = "./testdata/adv_image.png"
    image_size = 416
    cv2_image = cv2.imread(image_path)
    tensor_image = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(image_size)])(
            (cv2_image, np.zeros((1, 5))))[0].unsqueeze(0)
    return tensor_image


def save_image(image, image_path):
    pil_img = transforms.functional.to_pil_image(image[0])
    pil_img.save(image_path)


def make_box_image(image, boxes):
    pil_img = transforms.functional.to_pil_image(image[0])
    box_img = img.draw_boxes(pil_img, boxes)
    return box_img


def make_annotation_image(image, detections):
    pil_img = transforms.functional.to_pil_image(image[0])
    datasets_class_names_path = "./coco2014/coco.names"
    class_names = coco.load_class_names(datasets_class_names_path)

    ann_img = img.draw_annotations(pil_img, detections.xyxy,
                                   detections.class_labels, detections.confidences, class_names)
    return ann_img


def init_tensorboard(name=None):
    logdir = 'testdata/tbx/'
    # subprocess.Popen(['tensorboard', f'--logdir={logdir}'])
    time_str = time.strftime("%Y%m%d-%H%M%S")
    if name is not None:
        return SummaryWriter(f'{logdir}{time_str}_{name}')
    else:
        return SummaryWriter(f'{logdir}{time_str}')


def train_adversarial_image(orig_img, tbx_writer=None):
    epoch = 250  # T in paper default 250
    t_iter = 0  # t in paper (iterator)
    psnr_threshold = 0

    is_monitor_mode = (tbx_writer is not None)

    if torch.cuda.is_available():
        ground_truth_image = orig_img.to(
            device='cuda:0', dtype=torch.float)
        adv_image = ground_truth_image.detach()
        adv_image.requires_grad = True

    with torch.no_grad():
        # 素の画像を物体検出器にかけた時の出力をground truthとする
        gt_yolo_out = yolo_util.detect(adv_image)
        gt_nms_out = yolo_util.nms(gt_yolo_out)

        if gt_nms_out[0].nelement() == 0:
            # 元画像の検出がない場合は敵対的画像を生成できない
            return adv_image

        ground_truthes = yolo_util.detections_ground_truth(gt_nms_out[0])

        # ground truthesをクラスタリング
        group_labels = clustering.object_grouping(
            ground_truthes.xywh.to('cpu').detach().numpy().copy())
        ground_truthes.set_group_info(torch.from_numpy(
            group_labels.astype(np.float32)).clone().to(ground_truth_image.device))

    n_b = 3  # 論文内で定められたパッチ生成枚数を指定するためのパラメータ
    background_patch_boxes = torch.zeros(
        (ground_truthes.total_group*n_b, 4), device=adv_image.device)

    model = yolo.load_model(
        "weights/yolov3.cfg",
        "weights/yolov3.weights")
    model.eval()

    optimizer = optim.Adam([adv_image])
    # 勾配計算失敗時のデバッグ用
    torch.autograd.set_detect_anomaly(True)

    for t_iter in tqdm(range(epoch), leave=is_monitor_mode):

        adv_image.requires_grad = True

        # t回目のパッチ適用画像から物体検出する
        output = model(adv_image)
        detections = yolo_util.detections_loss(output[0], is_nms=False)
        # nms_out = yolo_util.nms(output)
        # detections = yolo_util.detections_loss(nms_out[0])
        # if nms_out[0].nelement() == 0:
        #     # 検出がない場合は終了
        #     return adv_image

        tpc_loss, tps_loss, fpc_loss, end_flag = total_loss(
            detections, ground_truthes, background_patch_boxes, adv_image.shape[2:4])
        loss = tpc_loss+tps_loss+fpc_loss

        optimizer.zero_grad()
        loss.backward()

        gradient_image = adv_image.grad

        with torch.no_grad():

            if t_iter == 0:
                # ループの最初にのみ実行
                # パッチ領域を決定する
                background_patch_boxes = pf.initial_background_patches(
                    ground_truthes, gradient_image).reshape((ground_truthes.total_group*n_b, 4))
            else:
                # パッチ領域を拡大する（縮小はしない）
                background_patch_boxes = pf.expanded_background_patches(
                    background_patch_boxes, gradient_image)

            # 勾配画像をパッチ領域の形に切り出す
            perturbated_image = pf.perturbation_in_background_patches(
                gradient_image, background_patch_boxes)
            # make_box_image(perturbated_image, background_patch_boxes)
            # パッチの正規化
            nomalized_perturbated_image = pf.perturbation_normalization(
                perturbated_image)
            # make_box_image(perturbated_image, background_patch_boxes)
            # adv_image-perturbated_imageの計算結果を[0,255]にクリップする
            adv_image = pf.update_i_with_pixel_clipping(
                adv_image, nomalized_perturbated_image)

            # psnr評価用の画像を切り出す
            psnr_truth_image = pf.perturbation_in_background_patches(
                ground_truth_image, background_patch_boxes).detach().cpu().numpy()
            psnr_eval_image = pf.perturbation_in_background_patches(
                adv_image, background_patch_boxes).detach().cpu().numpy()
            psnr = peak_signal_noise_ratio(psnr_truth_image, psnr_eval_image)

            if tbx_writer is not None:
                tbx_writer.add_scalar(
                    "total_loss", loss, t_iter)
                tbx_writer.add_scalar(
                    "tpc_loss", tpc_loss, t_iter)
                tbx_writer.add_scalar(
                    "tps_loss", tps_loss, t_iter)
                tbx_writer.add_scalar(
                    "fpc_loss", fpc_loss, t_iter)
                tbx_writer.add_scalar(
                    "psnr", psnr, t_iter)

                if t_iter % 10 == 0:

                    nms_out = yolo_util.nms(output)
                    nms_detections = yolo_util.detections_loss(nms_out[0])

                    det_image = transforms.functional.to_tensor(make_annotation_image(
                        adv_image, nms_detections))
                    tbx_writer.add_image(
                        "adversarial_image", det_image, t_iter)

                    bp_image = transforms.functional.to_tensor(make_box_image(
                        nomalized_perturbated_image, background_patch_boxes))
                    tbx_writer.add_image(
                        "background_patch_boxes", bp_image, t_iter)

            if ((psnr < psnr_threshold)
                    or (end_flag)):
                # psnrが閾値以下
                # もしくは損失計算時に条件を満たした場合(zの要素がすべて0の場合)
                break

    return adv_image.clone().cpu()


def main():
    arg_parser = argparse.ArgumentParser(
        description="generate adversarial image")
    arg_parser.add_argument("-m", "--mode", type=str,
                            default="monitor", help="Select execution mode")
    arg_parser.add_argument("-d", "--description", type=str,
                            default="", help="Description to be included in the data")
    args = arg_parser.parse_args()

    mode = args.mode
    description = args.description

    time_str = time.strftime("%Y%m%d_%H%M%S")
    print(f'start: {time_str}')
    output_dir = f'./testdata/{mode}/{time_str}{description}/'
    os.makedirs(output_dir)

    match mode:
        case "monitor":

            input_image_path = "./data/dog.jpg"
            image = get_image_from_file(input_image_path)

            tbx_writer = SummaryWriter(output_dir)
            adv_image = train_adversarial_image(image, tbx_writer)
            tbx_writer.close()

            output_image_path = output_dir + \
                f'adv_image_{time_str}.png'

            save_image(adv_image, output_image_path)
        case "evaluate":
            iterate_num = 1000

            train_path = "./coco2014/images/train2014/"
            train_annfile_path = "./coco2014/annotations/instances_train2014.json"

            yolo_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((416, 416)),
            ])

            train_set = CocoDetection(root=train_path,
                                      annFile=train_annfile_path, transform=yolo_transforms)
            train_loader = torch.utils.data.DataLoader(train_set)

            for image_idx, (image, _) in tqdm(enumerate(train_loader), total=iterate_num):
                if image_idx >= iterate_num:
                    break
                output_image_path = output_dir + \
                    f'{image_idx}_adv_image_{time_str}.png'
                adv_image = train_adversarial_image(image)
                save_image(adv_image, output_image_path)

        case _:
            raise Exception('modeが想定されていない値です')


if __name__ == "__main__":
    main()
