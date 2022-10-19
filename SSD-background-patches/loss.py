import torch
import math
from util.box import is_overlap


def calc_z(class_score: float, iou_score: float) -> int:
    """calc z param
    Args:
      class_score:
        攻撃対象クラスのクラススコア
      iou_score:
        攻撃対象クラスと対象としている検出のIoUスコア
    """
    # 正解領域は複数ある？
    class_score_threshold = 0.1
    iou_score_threshold = 0.5

    if (class_score > class_score_threshold) and (iou_score > iou_score_threshold):
        return 1
    else:
        return 0


def calc_r(iou_score, predict_boxes, target_boxes) -> int:
    """calc r param
    Args:
      iou_score:
        検出と背景パッチのIoUスコア
      predict_boxes:
        検出全て
      target_boxes:
        ground truthのボックス全て
    """
    iou_score_threshold = 0.1

    for predict_box in predict_boxes:
        for target_box in target_boxes:
            # 一つでも重なったらr=0
            if not is_overlap(predict_box, target_box):
                return 0

    if iou_score > iou_score_threshold:
        return 1
    else:
        return 0


def tpc(target_class_index: int, class_scores_list, iou_scores) -> float:
    """True Positive Class Loss
    Args:
      target_class_index:
        grand truth class index
        target_class_index as an index to class_scores
      class_scores_list:
        If there are c class scores and j detections, denoted by a list of j*c
      iou_scores_list:
        IOU score between detection and true bounding box, calculated for each j detections
    Returns:
      tpc_score(float)
    Raises:
    """
    tpc_score = 0.

    # Error if the number of elements in the class_scores and iou_scores are different
    for (class_scores, iou_score) in zip(class_scores_list, iou_scores):

        class_scores[target_class_index] = 0
        calc_target_index = torch.argmax(class_scores)

        z = calc_z(class_scores[calc_target_index].item(), iou_score.item())
        tpc_score += z*math.log(class_scores[calc_target_index].item())

    tpc_score *= -1.
    return tpc_score


def tps(target_class_index, class_scores_list, iou_scores, target_box, predict_boxes) -> float:
    """True Positive Shape Loss
    Args:
      target_class_index:
        攻撃対象クラスのclass_scores_listに対するインデックス
      class_scores_list:
        If there are c class scores and j detections, denoted by a list of j*c
      iou_scores_list:
        IOU score between detection and true bounding box, calculated for each j detections
      target_box:
        攻撃対象ボックスx1y1x2y2
      predict_boxes:
        全検出x1y1x2y2のリスト
    """
    tps_score = 0.

    for i, predict_box in enumerate(predict_boxes):

        box_diff = target_box-predict_box
        box_diff = torch.pow(box_diff, 2)

        class_scores_list[i, target_class_index] = 0
        calc_target_index = torch.argmax(class_scores_list[i, :])

        z = calc_z(
            class_scores_list[i, calc_target_index].item(), iou_scores[i].item())

        tps_score += z*torch.sum(box_diff).item()

    tps_score = math.exp(-1.*tps_score)
    return tps_score


def fpc(target_class_index, class_scores_list, iou_scores, target_boxes, predict_boxes) -> float:
    """False Positive Class Loss
    """
    fpc_score = 0.

    for class_scores, iou_score in zip(class_scores_list, iou_scores):

        class_scores[target_class_index] = 0
        calc_target_index = torch.argmax(class_scores)

        r = calc_r(iou_scores, target_boxes, predict_boxes)
        fpc_score += r*math.log(class_scores[calc_target_index].item())

    tpc_score *= -1.
    return fpc_score
