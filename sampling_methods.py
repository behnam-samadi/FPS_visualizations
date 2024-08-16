import numpy as np
from queue import PriorityQueue
import torch
import random
from utils import *
import sklearn
from frequency_domain_oprations import freq_based_sampling


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    src = torch.from_numpy(src)
    dst = torch.from_numpy(dst)
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def project_and_sort(xyz):
    projected_values = np.sum(xyz, 2)
    projected_values_ = np.sort(projected_values)
    order = np.argsort(projected_values)
    return projected_values_, order


def binary_search(projected, left, right, query):
    middle = int((left+right)/2)
    if right < left:
        return 0, left
    if query == projected[0,middle]:
        return 1, middle
    elif query < projected[0,middle]:
        return binary_search(projected, left, middle-1, query)
    elif query > projected[0,middle]:
        return binary_search(projected, middle+1, right, query)


def find_middle_candidate(projected, left, right):
    query = (projected[0,left] + projected[0, right])/2
    suc, res = binary_search(projected, left, right, query)
    if suc:
        return res, abs(projected[0, res] - projected[0, left])
    elif res == right + 1:
        return right, 0
    elif res == 0:
        return 0, 0
    else:
        if abs(projected[0, res-1] - query) <= abs(projected[0,res]- projected[0,right]):
            return res - 1, abs(projected[0,res-1] - projected[0,left])
        else:
            return res, abs(projected[0,res] - projected[0,right])


def farthest_point_sample_proposed(xyz, npoint):
    # proposed
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # implementing the proposed FPS is desinged for batch_size=1 (for inference)
    xyz = expand_point_cloud(xyz)
    device = xyz.device
    B, N, C = xyz.shape
    projected_values, order = project_and_sort(xyz)
    # print("Pre process time: ", pre_process_time)
    selected_points = list(np.expand_dims(np.random.randint(1, N - 1), axis=0))
    # print("projected_values.shape:", projected_values.shape)

    head_candidate_score = abs(projected_values[0, selected_points[0]] - projected_values[0, 0])
    tail_candidate_score = abs(projected_values[0, selected_points[0]] - projected_values[0, N - 1])
    candidates = PriorityQueue()
    candidates.put((-1 * head_candidate_score, 0, -2, selected_points[0]))
    candidates.put((-1 * tail_candidate_score, N - 1, selected_points[0], -1))
    for i in range(npoint - 1):
        _, next_selected, left_selected, right_selected = candidates.get()
        # selected_points = torch.cat((selected_points, torch.tensor([next_selected])), 0)
        selected_points.append(next_selected)
        # Adding the right-side candidate:
        if not (right_selected == -1 or right_selected == next_selected + 1):
            middle, score = find_middle_candidate(projected_values, next_selected, right_selected)
            candidates.put((-1 * score, middle, next_selected, right_selected))

        # Adding the left-side candidate:
        if not (left_selected == -2 or left_selected == next_selected - 1):
            middle, score = find_middle_candidate(projected_values, left_selected, next_selected)
            candidates.put((-1 * score, middle, left_selected, next_selected))

    centroids = np.zeros((1, npoint))
    centroids[0, 0:npoint] = order[0, selected_points]
    # TODO (important): re-arrange the selected points by the order tensor
    return centroids


def random_point_sample(xyz, npoint):
    # orig
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    xyz = torch.from_numpy(xyz)
    device = xyz.device
    B = 1
    N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    centroids = centroids.numpy()
    xyz = xyz.numpy()
    range_list = list(range(xyz.shape[0]))
    random_result = random.sample(range_list, npoint)
    # print(random_result)
    # print(np.squeeze)
    centroids = np.zeros((1, npoint))
    centroids[0, 0:npoint] = random_result
    return centroids


def random_sampling(xyz, npoint):
    # random
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    range_list = list(range(xyz.shape[1]))
    random_result = random.sample(range_list, npoint)
    centroids = np.zeros((1, npoint))
    centroids[0, 0:npoint] = random_result
    return centroids


def farthest_point_sample(xyz, npoint):
    # orig
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    if xyz.ndim == 2:
        xyz = np.expand_dims(xyz, axis = 0)
    xyz = torch.from_numpy(xyz)
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    centroids = centroids.numpy()
    distance = distance.numpy()
    farthest = farthest.numpy()
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    batch_indices = batch_indices.numpy()
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        dist = dist.numpy()
        #dist = torch.abs(torch.sum(xyz, -1) - torch.sum(centroid, -1))
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
        #farthest = farthest[1]
        #farthest = np.max(distance, -1)[1]

    return centroids

def distance_based_sampling(xyz, npoint):
    pc1 = pc_normalize(xyz)
    low_freq_rec1 = freq_based_sampling(pc1)
    # min_distances, arg_min_distances = find_distance_pc(pc2, low_freq_rec)
    distances_1 = sklearn.metrics.pairwise.euclidean_distances(pc1, low_freq_rec1)
    min_1 = np.min(distances_1, axis=1)
    sorted_1 = np.argsort(min_1)
    return sorted_1[0:npoint]

def distance_aware_farthest_point_sample(xyz, npoint):
    # orig
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """



    low_freq_rec = freq_based_sampling(xyz)
    low_freq_rec = pc_normalize(low_freq_rec)
    # min_distances, arg_min_distances = find_distance_pc(pc2, low_freq_rec)
    min_distances_2 = sklearn.metrics.pairwise.euclidean_distances(xyz, low_freq_rec)
    distances = np.min(min_distances_2, axis=1)
    if xyz.ndim == 2:
        xyz = np.expand_dims(xyz, axis = 0)

    xyz = torch.from_numpy(xyz)
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    centroids = centroids.numpy()
    distance = distance.numpy()
    farthest = farthest.numpy()
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    batch_indices = batch_indices.numpy()
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        dist = dist.numpy()
        #dist = torch.abs(torch.sum(xyz, -1) - torch.sum(centroid, -1))
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance - distances, -1)
        distances[farthest] = 1e10
        #farthest = farthest[1]
        #farthest = np.max(distance, -1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    xyz = expand_point_cloud(xyz)
    new_xyz = expand_point_cloud(new_xyz)
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def compute_coverage(grid_pc, orig_pc, centroids, radius = 0.5, nsample = 32):
    new_xyz = index_points(orig_pc, centroids)
    idx = query_ball_point(radius, nsample, grid_pc, new_xyz)
    idx = idx[0,:,:]
    idx = idx.flatten()
    idx = idx.tolist()
    idx = list(set(idx))
    return idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def find_distance_pc(pc1, pc2):
    #pc1 = expand_point_cloud(pc1)
    #pc2 = expand_point_cloud(pc2)
    #sqrdists = square_distance(pc1, pc2)
    #sqrdists = sqrdists.numpy()
    #min_distances = np.min(sqrdists, axis=0)
    #arg_min_distances = np.argmin(sqrdists, axis=0)
    #return min_distances, arg_min_distances

    #A = query_ball_point(10000, 1, pc1, pc2)


    min_distances = []
    arg_min_distances = []
    for i, point1 in enumerate(pc1):
        print(i)
        distances = []
        for point2 in pc2:
            distances.append(euclid_dis(point1, point2))
        distances = np.array(distances)
        min_distances.append(np.min(distances))
        arg_min_distances.append(np.argmin(distances))

    return np.array(min_distances), np.array(arg_min_distances)