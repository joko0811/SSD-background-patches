import torch

from omegaconf import DictConfig

from model.yolo_util import detections_ground_truth, detections_loss
from box.seek import find_nearest_box
from box.condition import are_overlap_list
from evaluation.detection import iou
from loss import li2019


def total_loss(detections: detections_loss, ground_truthes: detections_ground_truth, image_hw, config: DictConfig):
    """Returns the total loss
    Args:
      detections:
        yolo detections
        (x1, y1, x2, y2, conf, cls)
      ground_truthes:
        ground truth detections
        (x1, y1, x2, y2, conf, cls)
    """

    # 検出と一番近いground truth
    gt_nearest_idx = find_nearest_box(
        detections.xywh, ground_truthes.xywh)
    gt_box_nearest_dt = ground_truthes.xyxy[gt_nearest_idx]
    # detectionと、各detectionに一番近いground truthのiouスコアを算出
    dt_gt_iou_scores = iou(
        detections.xyxy, gt_box_nearest_dt)
    # dtのスコアから、gt_nearest_idxで指定されるground truthの属するクラスのみを抽出
    dt_scores_for_nearest_gt_label = detections.class_scores.gather(
        1, ground_truthes.class_labels[gt_nearest_idx, None]).squeeze()
    # 論文で提案された変数zを計算
    z = li2019.calc_z(dt_scores_for_nearest_gt_label,
                      dt_gt_iou_scores, config.calc_z)

    # 論文で提案された変数rを計算
    r = calc_r(detections.xyxy,
               ground_truthes.xyxy, config.calc_r)

    # 損失計算用の情報を積む
    detections.set_loss_info(gt_nearest_idx)

    # tpc、fpcの計算に必要なパラメータ計算
    max_class_scores = li2019.get_max_scores_without_correct_class(
        detections, ground_truthes)

    end_flag_z = (z.nonzero().size() == (0, 1))

    tpc_weight = config.tpc_weight  # default 0.1
    tps_weight = config.tps_weight  # default 1
    fpc_weight = config.fpc_weight  # default 1

    gt_z = li2019.mean_tps_calc_z(detections, ground_truthes, config.calc_z)
    tps_loss = li2019.mean_tps(
        gt_z, detections, ground_truthes, image_hw)

    tpc_loss = li2019.tpc(z, max_class_scores)*tpc_weight
    # tps_loss = normalized_tps(
    #     z, detections, ground_truthes, image_hw, config.nomalized_tps)*tps_weight
    fpc_loss = li2019.fpc(r, max_class_scores)*fpc_weight

    return (tpc_loss, tps_loss, fpc_loss, end_flag_z)


def calc_r(detection_boxes, ground_truth_boxes, config: DictConfig):
    """calc r param
    rの要素数はクラス数と等しい
    Args:
      iou_score:
        検出と背景パッチのIoUスコア
      predict_boxes:
        検出全て
      target_boxes:
        ground truthのボックス全て
    """
    overlap_flag = torch.logical_not(are_overlap_list(
        detection_boxes, ground_truth_boxes))
    r = overlap_flag
    return r.long()
