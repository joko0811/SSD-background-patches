from box import boxconv


class DetectionsBase:
    def __init__(self, conf_list, box_list, is_xywh=True):
        self.conf = conf_list
        if is_xywh:
            self.xywh = box_list
            self.xyxy = boxconv.xywh2xyxy(self.xywh)
        else:
            self.xyxy = box_list
            self.xywh = boxconv.xyxy2xywh(self.xyxy)

    def __len__(self):
        return len(self.conf)


class ObjectDetectionBase(DetectionsBase):
    def __init__(self, conf_list, box_list, class_score_list, is_xywh=True):
        super.__init__(conf_list, box_list, is_xywh)
        self.class_score = class_score_list
