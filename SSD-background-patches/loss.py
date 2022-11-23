import torch

from model.yolo_util import detections_ground_truth, detections_loss


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


def total_loss(detections: detections_loss, ground_truthes: detections_ground_truth):
    """Returns the total loss
    Args:
      detections:
        yolo detections
        (x1, y1, x2, y2, conf, cls)
      ground_truthes:
        ground truth detections
        (x1, y1, x2, y2, conf, cls)
    """

    # tpc、fpcの計算に必要なパラメータ計算
    max_class_scores = get_max_scores_without_correct_class(
        detections, ground_truthes)

    tpc_score = tpc(detections, max_class_scores)
    tps_score = tps(detections, ground_truthes)
    fpc_score = fpc(detections, max_class_scores)

    loss = tpc_score+tps_score+fpc_score
    return loss


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
