
def format_boxes(boxes, **kargs):
    box_str = ""

    for box_idx in range(boxes):
        x = boxes[box_idx, 0]
        y = boxes[box_idx, 0]
        w = boxes[box_idx, 0]
        h = boxes[box_idx, 0]
        box_str += (
            str(x.item()) + " " +
            str(y.item()) + " " +
            str(w.item()) + " " +
            str(h.item()) + "\n"
        )

    return box_str


def format_detections(detections, label_names):
    det_str = ""

    for det_idx in range(detections.total_det):
        label = label_names[detections.class_labels[det_idx]]
        conf = detections.confidences[det_idx]
        box = detections.xyxy[det_idx]

        det_str += (
            label + " " +
            str(conf.item()) + " " +
            str(int(box[0].item())) + " " +
            str(int(box[1].item())) + " " +
            str(int(box[2].item())) + " " +
            str(int(box[3].item())) + "\n"
        )

    return det_str


def format_yolo(detections, image_hw):
    det_str = ""

    for det_idx in range(detections.total_det):
        label_idx = detections.class_labels[det_idx]
        box = detections.xywh[det_idx]
        yolo_x = box[0].item()/image_hw[1]
        yolo_y = box[1].item()/image_hw[0]
        yolo_w = box[2].item()/image_hw[1]
        yolo_h = box[3].item()/image_hw[0]

        det_str += (
            str(label_idx.item()) + " " +
            str(yolo_x) + " " +
            str(yolo_y) + " " +
            str(yolo_w) + " " +
            str(yolo_h) + "\n"
        )

    return det_str
