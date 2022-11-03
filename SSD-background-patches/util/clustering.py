from sklearn.cluster import DBSCAN
import numpy as np


def object_grouping(detect_boxes):
    """initialize grouping in paper
    ひとまずDBSCAN
    """
    eps = 20
    clustering = DBSCAN(eps=eps, min_samples=2).fit(detect_boxes)
    return clustering.labels_


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
