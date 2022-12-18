import numpy as np

from tqdm import tqdm

import torch
from torchvision import transforms
from torchvision.datasets.coco import CocoDetection

import proposed_func as pf
from model import yolo, yolo_util
from util.clustering import object_grouping

from box import condition


def is_extractable_patch_for_image(detections, image):

    for i in range(detections.total_group):

        group_xyxy = detections.xyxy[detections.group_labels == i]
        group_xywh = detections.xywh[detections.group_labels == i]

        # 探索領域決定
        search_image, search_area = pf.calculate_search_area(
            image, group_xyxy)
        is_extractable_patch_for_group()

        # 最終的にグループ毎に3つ領域が選択できていたらサブセットに含める
    return
    # TODO:パッチ初期化時にパッチ領域を選択できる（＝十分な背景領域が存在する）画像を抽出するプログラムを書く


def is_extractable_patch_for_group(image, ignore_boxes, search_area):
    """
    """
    n_b = 3

    # ウインドウの高さ、幅の決定
    window_list_w, window_list_h = pf.calculate_window_wh()  # TODO

    extractable_patch_num = 0

    for window_w, window_h in zip(window_list_w, window_list_h):
        window_box_map = pf.create_box_map_of_original_image()  # TODO
        # [h,w,4]->[w*h,4]
        window_box_list = window_box_map.reshape(
            (window_box_map.shape[0]*window_box_map.shape[1], 4))
        overlap_table = condition.are_overlap_list(
            window_box_list, ignore_boxes)

        extractable_patch_num += overlap_table.nonzero(
        ).shape[0]

        if extractable_patch_num >= n_b:
            break

    if extractable_patch_num < n_b:
        # 規定の数だけ領域を選択できなかった場合
        return False

    return True


def gen_subset_with_sufficient_background_area(dataloader, model):
    """十分な背景領域があり、かつ検出が存在する画像を抽出したサブセットを作成する
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # グループ毎のパッチの数
    n_b = 3

    for (gt_image, _) in tqdm(dataloader):
        is_included_subset = True

        gt_image = gt_image.to(device=device, dtype=torch.float)

        # 検出・整形
        output = model(gt_image)
        nms_out = yolo_util.nms(output)
        detections = yolo_util.detections_ground_truth(nms_out[0])

        if nms_out[0].nelement() == 0:
            # 検出が存在しない画像はサブセットに含めない
            continue

        # クラスタリング
        group_labels = object_grouping(
            detections.xywh.detach().cpu().numpy())
        detections.set_group_info(torch.from_numpy(
            group_labels.astype(np.float32)).to(device))

    return


def main():

    yolo_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((416, 416)),
    ])
    train_path = "./coco2014/images/train2014/"
    train_annfile_path = "./coco2014/annotations/instances_train2014.json"
    coco_set = CocoDetection(root=train_path,
                             annFile=train_annfile_path, transform=yolo_transforms)
    coco_loader = torch.utils.data.DataLoader(coco_set)

    model = yolo.load_model(
        "weights/yolov3.cfg",
        "weights/yolov3.weights")
    model.eval()

    extract_images_with_sufficient_background_area(coco_loader, model)


if __name__ == "__main__":
    main()
