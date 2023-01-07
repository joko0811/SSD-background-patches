
def format_boxes(boxes):
    box_str = ""

    for i in range(boxes):
        box_str += format_box(i, boxes)+"\n"

    return box_str


def format_box(box_idx, boxes):
    x = boxes[box_idx, 0]
    y = boxes[box_idx, 0]
    w = boxes[box_idx, 0]
    h = boxes[box_idx, 0]
    return str(x.item())+" "+str(y.item()+" "+str(w.item())+" "+str(h.item()))


def save_detections(format_func, path, detections, class_names, image_wh):

    det_str = ""

    for i in range(detections.total_det):
        det_str += format_func(i, detections, class_names, image_wh)+"\n"

    with open(path, mode='w') as f:
        f.write(det_str)


def format_detections(det_idx, detections, label_names, image_hw):
    label = label_names[detections.class_labels[det_idx]]
    conf = detections.confidences[det_idx]
    box = detections.xyxy[det_idx]

    return label+" "+str(conf.item())+" "+str(int(box[0].item()))+" "+str(int(box[1].item()))+" "+str(int(box[2].item()))+" "+str(int(box[3].item()))


def format_yolo(det_idx, detections, label_names, image_hw):
    label_idx = detections.class_labels[det_idx]
    box = detections.xywh[det_idx]
    yolo_x = box[0].item()/image_hw[1]
    yolo_y = box[1].item()/image_hw[0]
    yolo_w = box[2].item()/image_hw[1]
    yolo_h = box[3].item()/image_hw[0]

    return str(label_idx.item())+" "+str(yolo_x)+" "+str(yolo_y)+" "+str(yolo_w)+" "+str(yolo_h)
