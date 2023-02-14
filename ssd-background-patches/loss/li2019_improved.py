import torch

from omegaconf import DictConfig

from model.s3fd_util import detections_s3fd_ground_truth, detections_s3fd_loss
from evaluation.detection import list_iou
from loss import proposed


def total_loss(detections: detections_s3fd_loss, ground_truthes: detections_s3fd_ground_truth, background_patch_boxes, image_hw, config: DictConfig):
    """Returns the total loss
    Args:
      detections:
        yolo detections
        (x1, y1, x2, y2, conf, cls)
      ground_truthes:
        ground truth detections
        (x1, y1, x2, y2, conf, cls)
    """

    dt_gt_iou_scores = list_iou(
        detections.xyxy, ground_truthes.xyxy)
    dt_bp_iou_scores = list_iou(detections.xyxy, background_patch_boxes)

    z = proposed.calc_z(dt_gt_iou_scores, config.calc_z)
    r = calc_r(dt_gt_iou_scores, dt_bp_iou_scores, config.calc_r)

    end_flag_z = (z.nonzero().size() == (0, 1))

    tpc_weight = config.tpc_weight  # default 0.1
    tps_weight = config.tps_weight  # default 1
    fpc_weight = config.fpc_weight  # default 1

    tpc_score = tpc_weight*proposed.tpc_loss(z, detections)
    tps_score = tps_weight * \
        proposed.tps_loss(z, detections, ground_truthes, image_hw)
    fpc_score = fpc_weight*proposed.fpc_loss(r, detections)
    return (tpc_score, tps_score, fpc_score, end_flag_z)


def calc_r(dt_gt_iou_scores, dt_bp_iou_scores, config):
    background_iou = config.background_iou  # default is 0
    patch_iou = config.patch_iou  # default is 0.1

    background_flag = (dt_gt_iou_scores <= background_iou).all(dim=1).long()
    patch_flag = (dt_bp_iou_scores > patch_iou).any(dim=1).long()

    r = background_flag*patch_flag
    return r
