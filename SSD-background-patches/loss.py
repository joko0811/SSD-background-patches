import torch

from model.yolo_util import detections_ground_truth, detections_loss
from box import condition, seek
import proposed_func as pf


def tpc(detections: detections_loss, max_class_scores):
    """True Positive Class Loss
    """

    # スコア計算
    tpc_score = -1 * torch.sum(detections.z * torch.log(max_class_scores))

    return tpc_score


def tps(detections: detections_loss, ground_truthes: detections_ground_truth):
    """True Positive Shape Loss
    """

    # (x-x)^2+(y-y)^2+(w-w)^2+(h-h)
    dist = (
        (detections.xywh-ground_truthes.xywh[detections.nearest_gt_idx])**2)

    tps_score = torch.exp(-1*torch.sum(detections.z*torch.sum(dist, dim=1)))

    return tps_score


def fpc(detections: detections_loss, max_class_scores):
    """False Positive Class Loss
    """

    # スコア計算
    fpc_score = -1 * torch.sum(detections.r * torch.log(max_class_scores))

    return fpc_score


def total_loss(detections: detections_loss, ground_truthes: detections_ground_truth, background_patch_boxes):
    """Returns the total loss
    Args:
      detections:
        yolo detections
        (x1, y1, x2, y2, conf, cls)
      ground_truthes:
        ground truth detections
        (x1, y1, x2, y2, conf, cls)
    """

    # nms_out = yolo_util.nms(output)
    # det_out = yolo_util.detections_loss(nms_out[0])
    # show_image(adv_image, det_out)

    # 検出と一番近いground truth
    gt_nearest_idx = seek.find_nearest_box(
        detections.xywh, ground_truthes.xywh)
    gt_box_nearest_dt = ground_truthes.xyxy[gt_nearest_idx]
    # detectionと、各detectionに一番近いground truthのiouスコアを算出
    dt_gt_iou_scores = condition.iou(
        detections.xyxy, gt_box_nearest_dt)
    # dtのスコアから、gt_nearest_idxで指定されるground truthの属するクラスのみを抽出
    dt_scores_for_nearest_gt_label = detections.class_scores.gather(
        1, ground_truthes.class_labels[gt_nearest_idx, None]).squeeze()
    # 論文で提案された変数zを計算
    z = pf.calc_z(dt_scores_for_nearest_gt_label, dt_gt_iou_scores)

    # 検出と一番近い背景パッチ
    bp_nearest_idx = seek.find_nearest_box(
        detections.xywh, background_patch_boxes)
    # detectionと、各detectionに一番近いground truthのiouスコアを算出
    bp_box_nearest_dt = background_patch_boxes[bp_nearest_idx]
    dt_bp_iou_scores = condition.iou(
        detections.xyxy, bp_box_nearest_dt)
    # 論文で提案された変数rを計算
    r = pf.calc_r(dt_bp_iou_scores, detections.xyxy,
                  ground_truthes.xyxy)

    # 損失計算用の情報を積む
    detections.set_loss_info(gt_nearest_idx, z, r)

    # tpc、fpcの計算に必要なパラメータ計算
    max_class_scores = get_max_scores_without_correct_class(
        detections, ground_truthes)

    tpc_loss = tpc(detections, max_class_scores)
    tps_loss = tps(detections, ground_truthes)
    fpc_loss = fpc(detections, max_class_scores)

    end_flag_z = (z.nonzero().size() == (0, 1))

    return (tpc_loss, tps_loss, fpc_loss, end_flag_z)


def get_max_scores_without_correct_class(detections: detections_loss, ground_truthes: detections_ground_truth):
    # detections.class_scoresで、計算対象外のクラスのクラススコアを0にしたtensorであるmasked_class_scoresを作成する
    # 検出ごとに一番近いground truthのクラスラベルを抽出
    gt_labels_for_nearest_dt = ground_truthes.class_labels[detections.nearest_gt_idx].unsqueeze(
        dim=1)
    # 次の行で参照するためだけのtensor
    ccs_src = torch.zeros(
        (detections.class_scores.shape[0], 1), device=detections.class_scores.device)
    # t番目の検出で計算対象外となるクラスはgt_labels_for_nearest_dt[t]
    # 計算対象外クラスが0、それ以外のクラスで1を取るマスク用配列
    mask_for_class_score = torch.ones(
        detections.class_scores.shape, device=detections.class_scores.device).scatter_(1, gt_labels_for_nearest_dt, ccs_src)
    # 計算対象外のクラスのクラススコアを0にした配列
    masked_class_scores = torch.mul(
        mask_for_class_score, detections.class_scores)

    # 正しいクラス以外のすべての全クラスのなかで最大のスコアを抽出
    max_class_scores = torch.max(masked_class_scores, dim=1)[0]

    return max_class_scores
