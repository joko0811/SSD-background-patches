import torch

from model.yolo_util import detections_ground_truth, detections_loss


class BPLoss(torch.nn.Module):
    def __init__(self):
        super(BPLoss, self).__init__()

    def forward(self, detections: detections_loss, ground_truthes: detections_ground_truth):

        tpc_score = self.tpc(detections, ground_truthes)
        tps_score = self.tps(detections, ground_truthes)
        fpc_score = self.fpc(detections, ground_truthes)

        loss = tpc_score+tps_score+fpc_score
        return loss

    def tpc(self, detections: detections_loss, ground_truthes: detections_ground_truth):
        """True Positive Class Loss
        """
        tpc_score = torch.tensor([0], dtype=torch.float,
                                 device=detections.data.device)

        gt_labels_for_nearest_dt = ground_truthes.class_labels[detections.nearest_gt_idx]
        calculated_class_score = detections.class_scores.detach()

        for i in range(calculated_class_score.shape[0]):
            ignore_idx = gt_labels_for_nearest_dt[i]
            calculated_class_score[i, ignore_idx] = 0

            tpc_score[0] += (detections.z[i] *
                             torch.log(torch.max(calculated_class_score[i])))

        tpc_score *= -1
        return tpc_score

    def tps(self, detections: detections_loss, ground_truthes: detections_ground_truth):
        """True Positive Shape Loss
        """
        tps_score = torch.tensor([0], dtype=torch.float,
                                 device=detections.data.device)

        # (x-x)^2+(y-y)^2+(w-w)^2+(h-h)
        dist = (
            (detections.xywh-ground_truthes.xywh[detections.nearest_gt_idx])**2)

        tps_score = torch.exp(-1*torch.sum(detections.z *
                              torch.sum(dist, dim=1)))

        return tps_score

    def fpc(self, detections: detections_loss, ground_truthes: detections_ground_truth):
        """False Positive Class Loss
        """
        fpc_score = torch.tensor([0], dtype=torch.float,
                                 device=detections.data.device)

        gt_labels_for_nearest_dt = ground_truthes.class_labels[detections.nearest_gt_idx]
        calculated_class_score = detections.class_scores.detach()

        for i in range(calculated_class_score.shape[0]):
            ignore_idx = gt_labels_for_nearest_dt[i]
            calculated_class_score[i, ignore_idx] = 0

            fpc_score[0] += (detections.r[i] *
                             torch.log(torch.max(calculated_class_score[i])))
        fpc_score *= -1
        return fpc_score


def tpc(detections: detections_loss, ground_truthes: detections_ground_truth):
    """True Positive Class Loss
    """
    tpc_score = torch.tensor([0], dtype=torch.float,
                             device=detections.data.device)

    gt_labels_for_nearest_dt = ground_truthes.class_labels[detections.nearest_gt_idx]
    calculated_class_score = detections.class_scores.detach()

    for i in range(calculated_class_score.shape[0]):
        ignore_idx = gt_labels_for_nearest_dt[i]
        calculated_class_score[i, ignore_idx] = 0

        tpc_score[0] += (detections.z[i] *
                         torch.log(torch.max(calculated_class_score[i])))

    tpc_score *= -1
    return tpc_score


def tps(detections: detections_loss, ground_truthes: detections_ground_truth):
    """True Positive Shape Loss
    """

    # (x-x)^2+(y-y)^2+(w-w)^2+(h-h)
    dist = (
        (detections.xywh-ground_truthes.xywh[detections.nearest_gt_idx])**2)

    tps_score = torch.exp(-1*torch.sum(detections.z*torch.sum(dist, dim=1)))

    return tps_score


def fpc(detections: detections_loss, ground_truthes: detections_ground_truth):
    """False Positive Class Loss
    """
    fpc_score = torch.tensor([0], dtype=torch.float,
                             device=detections.data.device)

    gt_labels_for_nearest_dt = ground_truthes.class_labels[detections.nearest_gt_idx]
    calculated_class_score = detections.class_scores.detach()

    for i in range(calculated_class_score.shape[0]):
        ignore_idx = gt_labels_for_nearest_dt[i]
        calculated_class_score[i, ignore_idx] = 0

        fpc_score[0] += (detections.r[i] *
                         torch.log(torch.max(calculated_class_score[i])))
    fpc_score *= -1
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
    tpc_score = tpc(detections, ground_truthes)
    tps_score = tps(detections, ground_truthes)
    fpc_score = fpc(detections, ground_truthes)

    loss = tpc_score+tps_score+fpc_score
    return loss


class TestNonObjectLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, detections, ground_truthes: detections_ground_truth, gt_nearest_idx, z, r):
        """Returns the total loss
        Args:
        detections:
            yolo detections
            (x1, y1, x2, y2, conf, cls)
        ground_truthes:
            ground truth detections
            (x1, y1, x2, y2, conf, cls)
        """
        tpc_score = self.test_tpc(
            detections, ground_truthes, z, gt_nearest_idx)
        tps_score = self.test_tps(
            detections, ground_truthes, z, gt_nearest_idx)
        fpc_score = self.test_fpc(
            detections, ground_truthes, r, gt_nearest_idx)

        loss = tpc_score+tps_score+fpc_score
        return loss

    def test_tpc(self, detections, ground_truthes: detections_ground_truth, z, gt_nearest_idx):
        """True Positive Class Loss
        """
        tpc_score = torch.tensor([0], dtype=torch.float,
                                 device=detections.device)

        gt_labels_for_nearest_dt = ground_truthes.class_labels[gt_nearest_idx]
        calculated_class_score = detections[:, 5:].detach()

        for i in range(calculated_class_score.shape[0]):
            ignore_idx = gt_labels_for_nearest_dt[i]
            calculated_class_score[i, ignore_idx] = 0

            tpc_score[0] += (z[i] *
                             torch.log(torch.max(calculated_class_score[i])))

        tpc_score *= -1
        return tpc_score

    def test_tps(self, detections, ground_truthes: detections_ground_truth, z, gt_nearest_idx):
        """True Positve Shape Loss
        """
        tps_score = torch.tensor([0], dtype=torch.float,
                                 device=detections.device)

        # (x-x)^2+(y-y)^2+(w-w)^2+(h-h)
        dist = (
            (detections[:, :4]-ground_truthes.xywh[gt_nearest_idx])**2)

        tps_score = torch.exp(-1*torch.sum(z*torch.sum(dist, dim=1)))

        return tps_score

    def test_fpc(self, detections, ground_truthes: detections_ground_truth, r, gt_nearest_idx):
        """False Positive Class Loss
        """
        fpc_score = torch.tensor([0], dtype=torch.float,
                                 device=detections.device)

        gt_labels_for_nearest_dt = ground_truthes.class_labels[gt_nearest_idx]
        calculated_class_score = detections[:, 5:].detach()

        for i in range(calculated_class_score.shape[0]):
            ignore_idx = gt_labels_for_nearest_dt[i]
            calculated_class_score[i, ignore_idx] = 0

            fpc_score[0] += (r[i] *
                             torch.log(torch.max(calculated_class_score[i])))
        fpc_score *= -1
        return fpc_score
