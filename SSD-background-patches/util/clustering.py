from sklearn.cluster import DBSCAN
import numpy as np


def object_grouping(boxes):
    """initialize grouping in paper
    ひとまずDBSCAN
    boxes=xywh
    """
    # 同じグループとみなすための最大距離
    eps = boxes.max()*0.2

    clustering = DBSCAN(eps=eps, min_samples=2).fit(boxes[:, :2])
    group_labels = clustering.labels_

    if (group_labels == -1).any():
        gl_max = group_labels.max()
        minus_iter = 1
        for i in range(len(group_labels)):
            if group_labels[i] == -1:
                group_labels[i] = gl_max+minus_iter
                minus_iter += 1

    return group_labels


def k_means(coordinates, k):
    """
    Args:
        coordinates:
            (x,y)
    """
    group_labeling = np.random.randint(0, k, coordinates.shape[0])
    prev_group_labeling = group_labeling.copy()

    center_gravities = [0, 0]*k

    while True:
        # 重心計算
        for i, c in enumerate(coordinates):
            group_num = group_labeling[i]

            center_gravities[group_num] += c
        center_gravities /= k

        # グループ再割り当て
        for i, c in enumerate(coordinates):
            for j, c_g in enumerate(center_gravities):
                if np.linalg.norm(c_g-c) < np.linalg.norm(center_gravities[group_labeling[i]]-c):
                    group_labeling[i] = j

        if prev_group_labeling == group_labeling:
            break
        else:
            prev_group_labeling = group_labeling.copy()

    return group_labeling
