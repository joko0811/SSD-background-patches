import torch

from omegaconf import DictConfig

from model.yolo_util import detections_ground_truth, detections_loss
from evaluation.detection import list_iou
from box.condition import are_overlap_list


def tpc():
    return


def tps():
    return


def mean_tps(z, detections: detections_loss, ground_truthes: detections_ground_truth, image_hw):
    """True Positive Shape Loss
    Args:
        z:
            M*N bainary param
        detections:
            N detection objects at that time
        ground_truthes:
            M detection objects of ground truth
    Return:
        scholar
    """
    calc_det = detections.xywh.unsqueeze(0)
    calc_gt = ground_truthes.xywh.unsqueeze(1)
    # M*N

    image_hw_gpu = torch.tensor(image_hw, device=calc_det.device)
    distance_div = torch.cat([image_hw_gpu, image_hw_gpu])
    distance = (torch.abs(calc_gt - calc_det) /
                distance_div).sum(dim=2)*z
    calc_distance = distance[distance > 0] if distance.nonzero(
    ).nelement() != 0 else torch.tensor(1e-5, device=distance.device)
    tps_score = torch.exp(-1*torch.mean(calc_distance))

    return tps_score


def fpc():
    return


def calc_z(detections: detections_loss, ground_truthes: detections_ground_truth, config: DictConfig):
    """calc z param
    The target element takes 1 when the following is true, and 0 otherwise
    (1) IoU is greater than the threshold value of 0.5
    (2) Class score equal to the Ground Truth label is greater than the threshold value of 0.1

    Args:
        detections:
            N detection objects at that time
        ground_truthes:
            M detection objects of ground truth
        config:
    Return:
        M*N binary
    """
    # set param
    iou_score_threshold = config.iou_score_threshold  # default 0.5
    class_score_threshold = config.class_score_threshold  # default 0.1

    # list of iou (N*M)
    iou_score_list = list_iou(ground_truthes.xyxy, detections.xyxy)
    # Determine if iou are above the threshold
    iou_score_flag = (iou_score_list > iou_score_threshold)

    sc_idx = ground_truthes.class_labels.tile(detections.total_det, 1)
    # list of class score equal to Ground Truth label (N*M)
    class_score_list = detections.class_scores.gather(1, sc_idx).T
    # Determine if class_score are above the threshold
    class_score_flag = (class_score_list > class_score_threshold)

    z = torch.logical_and(iou_score_flag, class_score_flag)

    return z.long()


def calc_r(detections: detections_loss, ground_truthes: detections_ground_truth, background_patch_boxes, config: DictConfig):
    """calc r param
    Args:
        detections:
            N detection objects at that time
        ground_truthes:
            detection objects of ground truth
        background_patch_boxes:
            background patches in xyxy format
        config:
    Return:
        N binary
    """
    # set param
    iou_score_threshold = config.iou_score_threshold  # default 0.1

    # list of iou
    iou_score_list = list_iou(background_patch_boxes, detections.xyxy)
    # Determine if iou are above the threshold
    iou_score_flag = (iou_score_list > iou_score_threshold).any(dim=1)

    # Detection that does not overlap with any ground truth
    # 1 element bool
    overlap_flag = torch.logical_not(are_overlap_list(
        detections.xyxy, ground_truthes.xyxy))
    r = torch.logical_and(iou_score_flag, overlap_flag)
    return r.long()
