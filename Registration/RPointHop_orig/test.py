import argparse
import os
import pickle
import time
import numpy as np
#import open3d as o3d
import sklearn
from Registration.RPointHop_orig import data_transforms
from Registration.RPointHop import modelnet40
from Registration.RPointHop_orig import rpointhop
from ploting import plot_style_2
from frequency_domain_operations import freq_based_sampling

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--initial_point', type=int, default=1024, help='Point Number [256/512/1024/2048]')
parser.add_argument('--model_dir', default='./model', help='Log dir [default: model]')
parser.add_argument('--num_point', default=[1024, 896, 768, 640], help='Point Number after down sampling')
parser.add_argument('--num_sample', default=[64, 32, 32, 32], help='KNN query number')
parser.add_argument('--source', default='./data/source_0.ply', help='Path to source point cloud')
parser.add_argument('--target', default='./data/target_0.ply', help='Path to target point cloud')
FLAGS = parser.parse_args()

initial_point = FLAGS.initial_point
num_point = FLAGS.num_point
num_sample = FLAGS.num_sample
source_path = FLAGS.source
target_path = FLAGS.target
MODEL_DIR = FLAGS.model_dir

def read_pc(file):

    #pcd_load = o3d.io.read_point_cloud(file)
    #pts = pcd_load.points
    #pt = np.asarray(pts)
    #if "source" in file:
        #pt = np.load("/home/behnam/temp/R-PointHop/source_0_corrupted.ply.npy")
    #if "target" in file:
        #pt = np.load("/home/behnam/temp/R-PointHop/target_0_corrupted.ply.npy")
    pt = np.load(file.split("./data/")[1]+".npy")
    #print(pt.shape)

    return pt



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


def register_orig(source_pc, target_pc):
    initial_point = source_pc.shape[0]
    target_e = np.expand_dims(target_pc, axis=0)
    source_e = np.expand_dims(source_pc, axis=0)
    data_c = np.concatenate((target_e, source_e), axis=0)

    with open(os.path.join("Registration/RPointHop_orig/"+MODEL_DIR, 'R-PointHop.pkl'), 'rb') as f:
        params = pickle.load(f)
    f.close()

    leaf_nodes, points = rpointhop.pointhop_pred(False, data_c, pca_params=params, n_newpoint=num_point,
                                                 n_sample=num_sample)
    features = np.array(leaf_nodes)
    features = np.reshape(features, (features.shape[0], features.shape[1], features.shape[2]))
    features = np.moveaxis(features, 0, 2)

    target_fea = features[0]
    source_fea = features[1]

    target_pts = points[0]
    source_pts = points[1]


    source_pc_rec = freq_based_sampling(source_pc)
    target_pc_rec = freq_based_sampling(target_pc)

    distances_source = sklearn.metrics.pairwise.euclidean_distances(source_pts, source_pc_rec)
    distances_target = sklearn.metrics.pairwise.euclidean_distances(target_pts, target_pc_rec)

    sorted_distances_source = np.sort(distances_source, axis = 1)[:,0:700]
    sorted_distances_target = np.sort(distances_target, axis = 1)[:, 0:700]
    result1= find_transform(target_fea, source_fea, target_pc, source_pc, target_pts, source_pts)
    result2 = find_transform(np.concatenate((target_fea, sorted_distances_target), axis = 1), np.concatenate((source_fea, sorted_distances_source), axis = 1), target_pc, source_pc, target_pts, source_pts)

    # result1 = find_transform(target_fea, source_fea, target_pc, source_pc, target_pts, source_pts)
    # result2 = find_transform(sorted_distances_target,
    #                          sorted_distances_source, target_pc, source_pc,
    #                          target_pts, source_pts)


    return result1, result2



def main():

    target_pc = read_pc(target_path)
    source_pc = read_pc(source_path)
    

    target_e = np.expand_dims(target_pc, axis=0)
    source_e = np.expand_dims(source_pc, axis=0)
    data_c = np.concatenate((target_e,source_e), axis=0)


    with open(os.path.join(MODEL_DIR, 'R-PointHop.pkl'), 'rb') as f:
        params = pickle.load(f)
    f.close()

    leaf_nodes, points = rpointhop.pointhop_pred(False, data_c, pca_params=params, n_newpoint=num_point, n_sample=num_sample)
    features = np.array(leaf_nodes)
    features = np.reshape(features,(features.shape[0],features.shape[1],features.shape[2]))
    features = np.moveaxis(features,0,2)
    
    target_fea = features[0]
    source_fea = features[1]

    target_pts = points[0]
    source_pts = points[1]

    distances = sklearn.metrics.pairwise.euclidean_distances(target_fea,source_fea)
    print(source_pts.shape)
    print(target_pts.shape)
    

    pred = np.argmin(distances,axis=0)
    dist_sort = np.sort(distances,axis=0)

    dist_ratio = dist_sort[0,:]/dist_sort[1,:]

    min_dist = np.min(distances,axis=0)
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

    x_mean = np.mean(source_pc,axis=0,keepdims=True)
    y_mean = np.mean(target_pc,axis=0,keepdims=True)

    data_x = data_x - x_mean
    data_y = data_y - y_mean

    cov = (data_y.T@data_x)
    u, s, v = np.linalg.svd(cov)
    R = v.T@u.T

    if (np.linalg.det(R) < 0):
        u, s, v = np.linalg.svd(cov)
        reflect = np.eye(3)
        reflect[2,2] = -1
        v = v.T@reflect
        R = v@u.T

    angle = data_transforms.matrix2euler(R, False)
    t = -R@y_mean.T+x_mean.T

    source_aligned = data_transforms.apply_inverse_transformation(source_pc, angle[2], angle[1], angle[0], t)
    np.save("result.npy", source_aligned)
    print(source_aligned.shape)


    # angle_pred = data_transforms.matrix2euler(R, True)
    print("resid ")
    exit(0)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pc)

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pc)

    source_aligned_pcd = o3d.geometry.PointCloud()
    source_aligned_pcd.points = o3d.utility.Vector3dVector(source_aligned)

    target_pcd.paint_uniform_color([220/255, 20/255, 60/255])
    source_pcd.paint_uniform_color([30/255, 144/255, 255/255])
    source_aligned_pcd.paint_uniform_color([30/255, 144/255, 255/255])
    
    o3d.visualization.draw_geometries([target_pcd,source_pcd])
    o3d.visualization.draw_geometries([target_pcd,source_aligned_pcd])
    
if __name__ == '__main__':
    main()
