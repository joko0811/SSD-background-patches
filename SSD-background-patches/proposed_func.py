import numpy as np

import torch

from util.clustering import object_grouping
import box


def calc_z(class_scores, iou_scores):
    """calc z param
    zの要素数はクラス数と等しい
    Args:
      class_score:
        GroundTruthesに対応する検出の攻撃対象クラスのクラススコア
      iou_score:
        Ground Truthesに対応する検出とGroundTruthesのIoUスコア
    """
    class_score_threshold = 0.1
    iou_score_threshold = 0.5

    z = torch.logical_and((class_scores > class_score_threshold),
                          (iou_scores > iou_score_threshold))
    return z.long()


def calc_r(iou_scores, detection_boxes, ground_truth_boxes):
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
    iou_score_threshold = 0.1
    iou_flag = iou_scores > iou_score_threshold
    overlap_flag = torch.tensor([box.condition.is_overlap_list(dt, ground_truth_boxes).all()
                                 for dt in detection_boxes], device=detection_boxes.device)
    r = torch.logical_and(iou_flag, overlap_flag)
    return r


def initial_background_patches(ground_truthes, gradient_image):
    initialize_size = 0.2
    aspect_rate = [1, 0.67, 0.75, 1.5, 1.33]
    largest_dist_late = 0.2
    # スライディングウインドウを用いた探索
    # 背景パッチとオブジェクト間の距離はオブジェクトボックスの最大辺の0.2倍
    group_label = object_grouping()
    return


def expanded_background_patches():
    return


def perturbation_in_background_patches():
    return


def perturbation_normalization():
    return


def update_i_with_pixel_clipping():
    return
