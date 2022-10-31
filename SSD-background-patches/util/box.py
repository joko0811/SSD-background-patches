import numpy as np


def xywh2x1y1(box):
    """xywh to lrtb(x1y1x2y2)
    """
    center_x, center_y, width, height = box

    left = center_x-(width/2)
    right = center_x+(width/2)
    top = center_y-(height/2)
    bottom = center_y+(height/2)
    return [left, right, top, bottom]


def lrtb2xywh(box):
    """lrtb(x1y1x2y2) to xywh
    """

    left, right, top, bottom = box

    x = (left+right)/2
    y = (top+bottom)/2
    w = abs(right-left)
    h = abs(bottom-top)
    return [x, y, w, h]


def is_overlap(boxA, boxB) -> bool:
    # box = lrtb (x1y1x2y2)
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    if boxA == boxB:
        return True
    elif (
        (ax1 <= bx1 and ax2 > bx1) or (ax1 >= bx1 and bx2 > ax1)
    ) and (
        (ay1 <= by1 and ay2 > by1) or (ay1 >= by1 and by2 > ay1)
    ):
        return True
    else:
        return False


def get_max_edge(box):
    # box = lrtb (x1y1x2y2)
    return max([abs(box[0]-box[2]), abs(box[1]-box[3])])


def kmeans(boxes, k, dist=np.median):
    """Calculates k-means clustering with the Intersection over Union (IoU) metric.
    Args:
        boxes:
            numpy array of shape (r, 2), where r is the number of rows
        k:
            number of clusters
        dist:
            distance function
    Returns:
        numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(
                boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


def iou(box, clusters):
    """Calculates the Intersection over Union (IoU) between a box and k clusters.
    Args:
        box:
            tuple or array, shifted to the origin (i. e. width and height)
        clusters:
            numpy array of shape (k, 2) where k is the number of clusters
    Returns:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_
