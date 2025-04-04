import torch

from omegaconf import DictConfig

from model.yolo_util import detections_yolo_ground_truth, detections_yolo_loss
from box.seek import find_nearest_box
from box.boxconv import are_overlap_list
from evaluation.detection import iou, list_iou


def tpc(z, max_class_scores):
    """True Positive Class Loss"""

    # スコア計算
    tpc_score = -1 * torch.sum(z * torch.log(max_class_scores))

    return tpc_score


def tps(
    z, detections: detections_yolo_loss, ground_truthes: detections_yolo_ground_truth
):
    """True Positive Shape Loss"""

    # (x-x)^2+(y-y)^2+(w-w)^2+(h-h)
    dist = (detections.xywh - ground_truthes.xywh[detections.nearest_gt_idx]) ** 2

    tps_score = torch.exp(-1 * torch.sum(z * torch.sum(dist, dim=1)))

    return tps_score


def fpc(r, max_class_scores):
    """False Positive Class Loss"""

    # スコア計算
    fpc_score = -1 * torch.sum(r * torch.log(max_class_scores))

    return fpc_score


def total_loss(
    detections: detections_yolo_loss,
    ground_truthes: detections_yolo_ground_truth,
    background_patch_boxes,
    image_hw,
    config: DictConfig,
):
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
    gt_nearest_idx = find_nearest_box(detections.xywh, ground_truthes.xywh)
    gt_box_nearest_dt = ground_truthes.xyxy[gt_nearest_idx]
    # detectionと、各detectionに一番近いground truthのiouスコアを算出
    dt_gt_iou_scores = iou(detections.xyxy, gt_box_nearest_dt)
    # dtのスコアから、gt_nearest_idxで指定されるground truthの属するクラスのみを抽出
    dt_scores_for_nearest_gt_label = detections.class_scores.gather(
        1, ground_truthes.class_labels[gt_nearest_idx, None]
    ).squeeze()
    # 論文で提案された変数zを計算
    z = calc_z(dt_scores_for_nearest_gt_label, dt_gt_iou_scores, config.calc_z)

    # 検出と一番近い背景パッチ
    bp_nearest_idx = find_nearest_box(detections.xywh, background_patch_boxes)
    # detectionと、各detectionに一番近いground truthのiouスコアを算出
    bp_box_nearest_dt = background_patch_boxes[bp_nearest_idx]
    dt_bp_iou_scores = iou(detections.xyxy, bp_box_nearest_dt)
    # 論文で提案された変数rを計算
    r = calc_r(dt_bp_iou_scores, detections.xyxy, ground_truthes.xyxy, config.calc_r)

    # 損失計算用の情報を積む
    detections.set_loss_info(gt_nearest_idx)

    # tpc、fpcの計算に必要なパラメータ計算
    max_class_scores = get_max_scores_without_correct_class(detections, ground_truthes)

    end_flag_z = z.nonzero().size() == (0, 1)

    tpc_weight = config.tpc_weight  # default 0.1
    tps_weight = config.tps_weight  # default 1
    fpc_weight = config.fpc_weight  # default 1

    gt_z = mean_tps_calc_z(detections, ground_truthes, config.calc_z)
    tps_loss = mean_tps(gt_z, detections, ground_truthes, image_hw)

    tpc_loss = tpc(z, max_class_scores) * tpc_weight
    # tps_loss = normalized_tps(
    #     z, detections, ground_truthes, image_hw, config.nomalized_tps)*tps_weight
    fpc_loss = fpc(r, max_class_scores) * fpc_weight

    return (tpc_loss, tps_loss, fpc_loss, end_flag_z)


def calc_z(class_scores, iou_scores, config: DictConfig):
    """calc z param
    zの要素数はクラス数と等しい
    Args:
      class_score:
        GroundTruthesに対応する検出の攻撃対象クラスのクラススコア
      iou_score:
        Ground Truthesに対応する検出とGroundTruthesのIoUスコア
    """
    class_score_threshold = config.class_score_threshold  # default 0.1
    iou_score_threshold = config.iou_score_threshold  # default 0.5

    z = torch.logical_and(
        (class_scores > class_score_threshold), (iou_scores > iou_score_threshold)
    )
    return z.long()


def calc_r(iou_scores, detection_boxes, ground_truth_boxes, config: DictConfig):
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
    iou_score_threshold = config.iou_score_threshold  # default 0.1
    iou_flag = iou_scores > iou_score_threshold
    overlap_flag = torch.logical_not(
        are_overlap_list(detection_boxes, ground_truth_boxes)
    )
    r = torch.logical_and(iou_flag, overlap_flag)
    return r.long()


def get_max_scores_without_correct_class(
    detections: detections_yolo_loss, ground_truthes: detections_yolo_ground_truth
):
    # detections.class_scoresで、計算対象外のクラスのクラススコアを0にしたtensorであるmasked_class_scoresを作成する
    # 検出ごとに一番近いground truthのクラスラベルを抽出
    gt_labels_for_nearest_dt = ground_truthes.class_labels[
        detections.nearest_gt_idx
    ].unsqueeze(dim=1)
    # 次の行で参照するためだけのtensor
    ccs_src = torch.zeros(
        (detections.class_scores.shape[0], 1), device=detections.class_scores.device
    )
    # t番目の検出で計算対象外となるクラスはgt_labels_for_nearest_dt[t]
    # 計算対象外クラスが0、それ以外のクラスで1を取るマスク用配列
    mask_for_class_score = torch.ones(
        detections.class_scores.shape, device=detections.class_scores.device
    ).scatter_(1, gt_labels_for_nearest_dt, ccs_src)
    # 計算対象外のクラスのクラススコアを0にした配列
    masked_class_scores = torch.mul(mask_for_class_score, detections.class_scores)

    # 正しいクラス以外のすべての全クラスのなかで最大のスコアを抽出
    max_class_scores, _ = torch.max(masked_class_scores, dim=1)

    # 損失計算時にmax_class_scoresの対数を取る
    # 対数計算結果でnanが発生するのを防止するため、max_class_scoresで値が0の要素は極小値と置換する
    zero_indexes = (max_class_scores == 0).nonzero()
    nan_proofed_max_class_scores = max_class_scores.scatter_(
        0,
        zero_indexes,
        torch.ones(zero_indexes.shape, device=zero_indexes.device) * 1e-9,
    )

    return nan_proofed_max_class_scores


# ---The following is for verification---


def normalized_tps(
    z,
    detections: detections_yolo_loss,
    ground_truthes: detections_yolo_ground_truth,
    image_hw,
    config: DictConfig,
):
    """True Positive Shape Loss"""
    # xyは画像サイズで割る　[0,1]区間に正規化
    # whも
    # (x-x)^2+(y-y)^2+(w-w)^2+(h-h)

    # dist_div = torch.tensor([image_hw[1], image_hw[0], image_hw[1]/hw_weight, image_hw[0]/hw_weight],

    hw_weight = config.hw_weight  # default 0.1
    gt_xywh_for_nearest_dt = ground_truthes.xywh[detections.nearest_gt_idx]
    dist_div = torch.cat(
        [gt_xywh_for_nearest_dt[:, :2], gt_xywh_for_nearest_dt[:, 2:4] / hw_weight],
        dim=1,
    )
    dist = (
        (detections.xywh - ground_truthes.xywh[detections.nearest_gt_idx]) / dist_div
    ).pow(2)

    tps_score = torch.exp(-1 * torch.sum(torch.sum(dist[:, :2], dim=1)))

    return tps_score


def mean_tps(
    z,
    detections: detections_yolo_loss,
    ground_truthes: detections_yolo_ground_truth,
    image_hw,
):
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
    # distance = (torch.abs(calc_gt - calc_det) / distance_div)[..., :2].sum(dim=2)*z
    distance = (torch.abs(calc_gt - calc_det) / distance_div).sum(dim=2) * z
    calc_distance = (
        distance[distance > 0]
        if distance.nonzero().nelement() != 0
        else torch.tensor(1e-9, device=distance.device)
    )
    tps_score = torch.exp(-1 * torch.mean(calc_distance))

    return tps_score


def mean_tps_calc_z(
    detections: detections_yolo_loss,
    ground_truthes: detections_yolo_ground_truth,
    config: DictConfig,
):
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
    iou_score_flag = iou_score_list > iou_score_threshold

    sc_idx = ground_truthes.class_labels.tile(len(detections), 1)
    # list of class score equal to Ground Truth label (N*M)
    class_score_list = detections.class_scores.gather(1, sc_idx).T
    # Determine if class_score are above the threshold
    class_score_flag = class_score_list > class_score_threshold

    z = torch.logical_and(iou_score_flag, class_score_flag)

    return z.long()
