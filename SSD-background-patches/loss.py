import torch

from model.yolo_util import detections_ground_truth, detections_loss
from box.seek import find_nearest_box
from evaluation.detection import iou
from proposed_func import calc_r, calc_z


def tpc(z, detections: detections_loss, max_class_scores):
    """True Positive Class Loss
    """

    # スコア計算
    tpc_score = -1 * torch.sum(z * torch.log(max_class_scores))

    return tpc_score


def tps(z, detections: detections_loss, ground_truthes: detections_ground_truth):
    """True Positive Shape Loss
    """

    # (x-x)^2+(y-y)^2+(w-w)^2+(h-h)
    dist = (
        (detections.xywh-ground_truthes.xywh[detections.nearest_gt_idx])**2)

    tps_score = torch.exp(-1*torch.sum(z*torch.sum(dist, dim=1)))

    return tps_score


def normalized_tps(z, detections: detections_loss, ground_truthes: detections_ground_truth, image_hw):
    """True Positive Shape Loss
    """
    # xyは画像サイズで割る　[0,1]区間に正規化
    # whも
    # (x-x)^2+(y-y)^2+(w-w)^2+(h-h)

    # dist_div = torch.tensor([image_hw[1], image_hw[0], image_hw[1]/hw_weight, image_hw[0]/hw_weight],

    gt_xywh_for_nearest_dt = ground_truthes.xywh[detections.nearest_gt_idx]
    hw_weight = 0.1
    dist_div = torch.cat(
        [gt_xywh_for_nearest_dt[:, :2], gt_xywh_for_nearest_dt[:, 2:4]/hw_weight], dim=1)
    dist = (
        (detections.xywh-ground_truthes.xywh[detections.nearest_gt_idx])/dist_div).pow(2)

    tps_score = torch.exp(-1*torch.sum(torch.sum(dist[:, :2], dim=1)))

    return tps_score


def mean_tps(z, detections: detections_loss, ground_truthes: detections_ground_truth):
    """True Positive Shape Loss
    """
    dist = torch.abs(
        detections.xywh[:, :2]-ground_truthes.xywh[detections.nearest_gt_idx, :2]+1e-5)
    z_sum = z*torch.sum(dist, dim=1)

    unique_labels, labels_count = detections.nearest_gt_idx.unique(
        return_counts=True)
    # nearest_gt_idxに出現しないラベルを含めた除算用のラベル毎集計変数を作成
    labels_count_per_class = torch.ones(torch.max(
        unique_labels)+1, device=z_sum.device, dtype=labels_count.dtype).scatter(0, unique_labels, labels_count)
    # 一番近いGround Truthのインデックスでラベル付し、ラベルに属する検出のz_sumの平均をとる
    average_by_gt = torch.zeros(torch.max(unique_labels)+1, device=z_sum.device, dtype=z_sum.dtype).scatter_add(
        0, detections.nearest_gt_idx, z_sum)/labels_count_per_class

    tps_score = torch.exp(-1*torch.sum(average_by_gt))

    return tps_score


def fpc(r, detections: detections_loss, max_class_scores):
    """False Positive Class Loss
    """

    # スコア計算
    fpc_score = -1 * torch.sum(r * torch.log(max_class_scores))

    return fpc_score


def total_loss(detections: detections_loss, ground_truthes: detections_ground_truth, background_patch_boxes, image_hw):
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
    z = calc_z(dt_scores_for_nearest_gt_label, dt_gt_iou_scores)

    # 検出と一番近い背景パッチ
    bp_nearest_idx = find_nearest_box(
        detections.xywh, background_patch_boxes)
    # detectionと、各detectionに一番近いground truthのiouスコアを算出
    bp_box_nearest_dt = background_patch_boxes[bp_nearest_idx]
    dt_bp_iou_scores = iou(
        detections.xyxy, bp_box_nearest_dt)
    # 論文で提案された変数rを計算
    r = calc_r(dt_bp_iou_scores, detections.xyxy,
               ground_truthes.xyxy)

    # 損失計算用の情報を積む
    detections.set_loss_info(gt_nearest_idx)

    # tpc、fpcの計算に必要なパラメータ計算
    max_class_scores = get_max_scores_without_correct_class(
        detections, ground_truthes)

    end_flag_z = (z.nonzero().size() == (0, 1))

    tpc_loss = tpc(z, detections, max_class_scores)*0.1
    tps_loss = normalized_tps(z, detections, ground_truthes, image_hw)
    fpc_loss = fpc(r, detections, max_class_scores)

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
    max_class_scores, _ = torch.max(masked_class_scores, dim=1)

    # 損失計算時にmax_class_scoresの対数を取る
    # 対数計算結果でnanが発生するのを防止するため、max_class_scoresで値が0の要素は極小値と置換する
    zero_indexes = (max_class_scores == 0).nonzero()
    nan_proofed_max_class_scores = max_class_scores.scatter_(
        0, zero_indexes, torch.ones(zero_indexes.shape, device=zero_indexes.device) * 1e-5)

    return nan_proofed_max_class_scores
