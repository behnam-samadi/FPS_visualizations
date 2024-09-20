from frequency_domain_operations import freq_based_sampling
import sklearn
import numpy as np
from Registration.RPointHop_orig import data_transforms
from sampling_methods import farthest_point_sample


def find_transform(target_fea, source_fea, target_pc, source_pc, target_pts, source_pts):
    distances = sklearn.metrics.pairwise.euclidean_distances(target_fea, source_fea)
    min_score = np.min(distances, axis=0)
    score = np.sum(min_score)

    pred = np.argmin(distances, axis=0)
    dist_sort = np.sort(distances, axis=0)

    dist_ratio = dist_sort[0, :] / dist_sort[1, :]

    min_dist = np.min(distances, axis=0)
    ordered = np.argsort(min_dist)
    pred = pred[ordered[:384]]
    data_x = source_pts[ordered[:384]]
    dist_ratio = dist_ratio[ordered[:384]]

    dist_ratio_ord = np.argsort(dist_ratio)
    pred = pred[dist_ratio_ord[:256]]
    data_x = data_x[dist_ratio_ord[:256]]

    sort = []
    for i in range(256):
        sort.append(target_pts[pred[i]])
    data_y = np.array(sort)

    x_mean = np.mean(source_pc, axis=0, keepdims=True)
    y_mean = np.mean(target_pc, axis=0, keepdims=True)

    data_x = data_x - x_mean
    data_y = data_y - y_mean

    cov = (data_y.T @ data_x)
    u, s, v = np.linalg.svd(cov)
    R = v.T @ u.T

    if (np.linalg.det(R) < 0):
        u, s, v = np.linalg.svd(cov)
        reflect = np.eye(3)
        reflect[2, 2] = -1
        v = v.T @ reflect
        R = v @ u.T

    angle = data_transforms.matrix2euler(R, False)
    t = -R @ y_mean.T + x_mean.T

    source_aligned = data_transforms.apply_inverse_transformation(source_pc, angle[2], angle[1], angle[0], t)
    return source_aligned




def dis_to_rec_based_register(source_pc, target_pc):
    source_pc_rec = freq_based_sampling(source_pc)
    target_pc_rec = freq_based_sampling(target_pc)
    distances_source = sklearn.metrics.pairwise.euclidean_distances(source_pc, source_pc_rec)
    distances_target = sklearn.metrics.pairwise.euclidean_distances(target_pc, target_pc_rec)
    sorted_distances_source = np.sort(distances_source, axis=1)[:, 0:700]
    sorted_distances_target = np.sort(distances_target, axis=1)[:, 0:700]
    result = find_transform(sorted_distances_target,sorted_distances_source, target_pc, source_pc,target_pc, source_pc)
    return result






