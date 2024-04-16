import os

import numpy as np
from tqdm import tqdm

from box.boxio import parse_detections
from detection.detection_base import DetectionsBase
from detection.tp_fp_manager import TpFpManager


def classify(det_path, gt_path):
    tp_det = np.empty((0, 5))
    fp_det = np.empty((0, 5))
    fn_det = np.empty((0, 5))

    device = "cpu"
    tpfp_manager = TpFpManager(device=device)

    for root, dir, files in os.walk(det_path):
        for file in tqdm(files):
            det_file_path = os.path.join(det_path, file)
            gt_file_path = os.path.join(gt_path, file)

            if (not os.path.exists(det_file_path)) or (
                not os.path.exists(gt_file_path)
            ):
                continue

            det_conf, det_box = parse_detections(det_file_path)
            gt_conf, gt_box = parse_detections(gt_file_path)

            det = DetectionsBase(det_conf, det_box, is_xywh=False)
            gt = DetectionsBase(gt_conf, gt_box, is_xywh=False)

            _, tp, fp, fn = tpfp_manager.judge_detection(det, gt)
            tp_det = np.concatenate((tp_det, tp.numpy(force=True)))
            fp_det = np.concatenate((fp_det, fp.numpy(force=True)))
            fn_det = np.concatenate((fn_det, fn.numpy(force=True)))

    np.savetxt("tp_det.csv", tp_det, delimiter=",")
    np.savetxt("fp_det.csv", fp_det, delimiter=",")
    np.savetxt("fn_det.csv", fn_det, delimiter=",")


def main():
    det_path = "outputs/eval_background/2024-03-13_10-12-53/detections"
    gt_path = "datasets/faceforensics_pp/test/detection/"

    if not os.path.exists(det_path):
        print(det_path, "is not found.")

    if not os.path.exists(gt_path):
        print(gt_path, "is not found.")

    classify(det_path, gt_path)


if __name__ == "__main__":
    main()
