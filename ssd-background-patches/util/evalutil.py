import os
from model import yolo_util


def gen_detection_path(image_path_list, detection_dir):
    detection_path_list = list()
    for image_path in image_path_list:
        # {path}/{to}/{image_name}.{extension}->{image_name}
        image_name = os.path.basename(image_path).split(".")[0] + ".txt"
        detection_path_list.append(os.path.join(detection_dir, image_name))

    return detection_path_list


def save_detection_text(image_list, path_list, model, format_func, optional=None):

    output = model(image_list)
    nms_out = yolo_util.nms(output)
    detection_list = yolo_util.make_detections_list(
        nms_out, yolo_util.detections_base)

    for detection, path in zip(detection_list, path_list):
        det_str = format_func(detection, optional)
        with open(path, mode="w") as f:
            f.write(det_str)
