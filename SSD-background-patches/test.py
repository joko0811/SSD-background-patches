import numpy as np
import cv2

import subprocess
import time

import torch
import torch.optim as optim
from torchvision.datasets.coco import CocoDetection
from torchvision import transforms

from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
from skimage.metrics import peak_signal_noise_ratio

from tensorboardX import SummaryWriter
from torchviz import make_dot
from tqdm import tqdm

from util import img, clustering
from model import yolo, yolo_util
from dataset import coco
from loss import total_loss
import proposed_func as pf


def get_image_from_dataset():
    train_path = "./coco2014/images/train2014/"
    train_annfile_path = "./coco2014/annotations/instances_train2014.json"

    coco_train = CocoDetection(root=train_path,
                               annFile=train_annfile_path, transform=transforms.ToTensor())
    img, _ = coco_train[0:1]
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    return img


def get_image_from_file(image_path):
    # image_path = "./testdata/adv_image.png"
    image_size = 416
    image = cv2.imread(image_path)
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(image_size)])(
            (image, np.zeros((1, 5))))[0].unsqueeze(0)
    return input_img


def make_annotation_image(orig_img, detections):
    pil_img = transforms.functional.to_pil_image(orig_img[0])
    datasets_class_names_path = "./coco2014/coco.names"
    class_names = coco.load_class_names(datasets_class_names_path)

    ann_img = img.draw_annotations(pil_img, detections.xyxy,
                                   detections.class_labels, detections.confidences, class_names)
    return ann_img


def make_box_image(image, boxes):
    pil_img = transforms.functional.to_pil_image(image[0])
    tmp_img = img.draw_boxes(pil_img, boxes)
    return tmp_img


def save_image(image, image_path):
    pil_img = transforms.functional.to_pil_image(image[0])
    pil_img.save(image_path)


def test_detect(input_path, output_path):
    image = get_image_from_file(input_path)
    if torch.cuda.is_available():
        gpu_image = image.to(
            device='cuda:0', dtype=torch.float)
    yolo_out = yolo_util.detect(gpu_image)
    nms_out = yolo_util.nms(yolo_out)
    detections = yolo_util.detections_base(nms_out[0])
    anno_image = make_annotation_image(image, detections)
    anno_image.save(output_path)


def init_tensorboard(name=None):
    logdir = 'testdata/tbx/'
    subprocess.Popen(['tensorboard', f'--logdir={logdir}'])
    time_str = time.strftime("%Y%m%d-%H%M%S")
    if name is not None:
        return SummaryWriter(f'{logdir}{time_str}_{name}')
    else:
        return SummaryWriter(f'{logdir}{time_str}')


def test_loss(orig_img):

    epoch = 250  # T in paper
    t_iter = 0  # t in paper (iterator)
    psnr_threshold = 0

    # 勾配計算失敗時のデバッグ用
    torch.autograd.set_detect_anomaly(True)

    if torch.cuda.is_available():
        ground_truth_image = orig_img.to(
            device='cuda:0', dtype=torch.float)
        adv_image = ground_truth_image.detach()
        adv_image.requires_grad = True

    with torch.no_grad():
        # 素の画像を物体検出器にかけた時の出力をground truthとする
        gt_yolo_out = yolo_util.detect(adv_image)
        gt_nms_out = yolo_util.nms(gt_yolo_out)
        ground_truthes = yolo_util.detections_ground_truth(gt_nms_out[0])

        # ground truthesをクラスタリング
        group_labels = clustering.object_grouping(
            ground_truthes.xywh.to('cpu').detach().numpy().copy())
        ground_truthes.set_group_info(torch.from_numpy(
            group_labels.astype(np.float32)).clone().to(ground_truth_image.device))

    n_b = 3  # 論文内で定められたパッチ生成枚数を指定するためのパラメータ
    background_patch_boxes = torch.zeros(
        (ground_truthes.total_group*n_b, 4), device=adv_image.device)

    tbx_writer = init_tensorboard()

    optimizer = optim.Adam([adv_image])

    model = yolo.load_model(
        "weights/yolov3.cfg",
        "weights/yolov3.weights")
    model.eval()
    torch.autograd.set_detect_anomaly(True)

    for t_iter in tqdm(range(epoch)):

        adv_image.requires_grad = True

        # t回目のパッチ適用画像から物体検出する
        output = model(adv_image)
        detections = yolo_util.detections_loss(output[0], is_nms=False)

        tpc_loss, tps_loss, fpc_loss, end_flag = total_loss(
            detections, ground_truthes, background_patch_boxes)
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
            perturbated_image = pf.perturbation_normalization(
                perturbated_image)
            # make_box_image(perturbated_image, background_patch_boxes)
            # adv_image-perturbated_imageの計算結果を[0,255]にクリップする
            adv_image = pf.update_i_with_pixel_clipping(
                adv_image, perturbated_image)

            # psnr評価用の画像を切り出す
            psnr_truth_image = pf.perturbation_in_background_patches(
                ground_truth_image, background_patch_boxes).detach().cpu().numpy()
            psnr_eval_image = pf.perturbation_in_background_patches(
                adv_image, background_patch_boxes).detach().cpu().numpy()
            psnr = peak_signal_noise_ratio(psnr_truth_image, psnr_eval_image)

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
                tbx_writer.add_image("adversarial_image",
                                     adv_image[0], t_iter)

                bp_image = transforms.functional.to_tensor(make_box_image(
                    perturbated_image, background_patch_boxes))
                tbx_writer.add_image(
                    "background_patch_boxes", bp_image, t_iter)

            if ((psnr < psnr_threshold)
                    or (end_flag)):
                # psnrが閾値以下もしくは損失計算時に条件を満たした場合(zの要素がすべて0の場合)ループを抜ける
                break

    print("success!")
    return adv_image


def run():

    input_path = "./testdata/adv_image.png"
    output_path = "./testdata/det_image.png"
    test_detect(input_path, output_path)

    # input_path = "./data/dog.jpg"
    # output_path = "./testdata/adv_image.png"
    # get_image_from_file(input_path)
    # adv_image=test_detect(img)
    # save_image(adv_image,output_path)


def main():
    run()


if __name__ == '__main__':
    main()
