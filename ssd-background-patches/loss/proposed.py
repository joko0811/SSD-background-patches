import torch

from omegaconf import DictConfig

from model.s3fd_util import detections_s3fd_loss, detections_s3fd
from evaluation.detection import list_iou
from box.seek import find_nearest_box
from loss.sharif2016 import tv_loss


def total_loss(detections: detections_s3fd_loss, ground_truthes: detections_s3fd, image_list, config: DictConfig):
    """Returns the total loss
    Args:
      detections:
        yolo detections
        (x1, y1, x2, y2, conf, cls)
      ground_truthes:
        ground truth detections
        (x1, y1, x2, y2, conf, cls)
    """

    # calc parameter z
    dt_gt_iou_scores = list_iou(
        detections.xyxy, ground_truthes.xyxy)
    z = calc_z(dt_gt_iou_scores, config.calc_z)
    bar_z = (z == 0).long()

    tpc_weight = config.tpc_weight  # default 0.1
    tps_weight = config.tps_weight  # default 1
    fpc_weight = config.fpc_weight  # default 1
    tv_weight = config.tv_weight

    tpc_score = tpc_weight*tpc_loss(z, detections)
    tps_score = tps_weight * \
        tps_loss(z, detections, ground_truthes, image_list.shape[-2:])
    fpc_score = fpc_weight*fpc_loss(bar_z, detections)
    tv_score = tv_weight*tv_loss(image_list)

    return (tpc_score, tps_score, fpc_score, tv_score)


def calc_z(dt_gt_iou_scores, config):
    iou_threshold = config.iou_threshold  # default 0.5
    z = (dt_gt_iou_scores >= iou_threshold).any(dim=1).long()
    return z


def tpc_loss(z, detections):
    tpc_score = -1*(torch.sum(z*torch.log(1-detections.conf+1e-5)))
    return tpc_score


def tps_loss(z, detections, ground_truthes, image_hw):
    gt_nearest_idx = find_nearest_box(
        detections.xywh, ground_truthes.xywh)
    gt_xywh_nearest_dt = ground_truthes.xywh[gt_nearest_idx]

    image_hw_gpu = torch.tensor(image_hw, device=z.device)
    distance_div = torch.cat([image_hw_gpu, image_hw_gpu])
    # distance = (torch.abs(calc_gt - calc_det) / distance_div)[..., :2].sum(dim=2)*z
    distance = (torch.abs(gt_xywh_nearest_dt -
                detections.xywh)[..., :2]).sum(dim=1)*z
    calc_distance = distance if distance.nonzero(
    ).nelement() != 0 else torch.tensor(1e-5, device=distance.device)
    tps_score = torch.exp(-1*torch.mean(calc_distance))

    return tps_score


def fpc_loss(bar_z, detections):
    fpc_score = -1 * \
        (torch.sum(bar_z*torch.log(detections.conf)))
    return fpc_score
