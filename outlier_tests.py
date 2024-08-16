import numpy as np

from utils import *
from ploting import *
from outlier import add_outlier_to_pc
from sampling_methods import *
import sklearn
from Registration.RPointHop.test import register
from Registration.RPointHop_orig.test import register_orig
from frequency_domain_operations import freq_based_sampling
import time
import h5py
from Registration.RPointHop.modelnet40 import data_load
from make_registration_data import make_target_frame
import os
from dataset.make_registration_dataset import create_dataset





def compare_methods():
    frames_address = "dataset/samples/"
    arr = os.listdir(frames_address)
    for file in arr:
        print(file)
        address = frames_address + file
        sample = np.load(address)
        pc = sample[:, 0, :]
        pc2 = sample[:, 1, :]
        pc_ = sample[:, 2, :]
        pc2_ = sample[:, 3, :]
        result1 = register(pc, pc2)
        result2 = register_orig(pc, pc2)
        result3 = register(pc_, pc2_)
        result4 = register_orig(pc_, pc2_)

        distances = sklearn.metrics.pairwise.euclidean_distances(pc2, result1)
        score = np.sum(np.min(distances ** 2, axis=0) ** 2)
        print(rmse(result1, pc2))
        #print("score: ", score)

        distances = sklearn.metrics.pairwise.euclidean_distances(pc2, result2)
        score = np.sum(np.min(distances ** 2, axis=0) ** 2)
        print(rmse(result2, pc2))
        #print("score: ", score)

        distances = sklearn.metrics.pairwise.euclidean_distances(pc2_, result3)
        score = np.sum(np.min(distances ** 2, axis=0) ** 2)
        print(rmse(result3, pc2_))
        #print("score: ", score)

        distances = sklearn.metrics.pairwise.euclidean_distances(pc2_, result4)
        score = np.sum(np.min(distances, axis=0) ** 2)
        print(rmse(result4, pc2_))
        #print("score: ", score)


def visualize_distance_based_sampling():
    #Tests on visualizing distance aware sampling
    sample = np.load("/home/behnam/phd/research/frequency_domain/pythonProject/dataset/samples/airplane_0009.npy")
    pc1 = sample[:,0,:]
    pc2 = sample[:,1,:]
    #pc1 = np.load("/home/behnam/temp/R-PointHop/source_0.ply.npy")
    #pc2 = np.load("/home/behnam/temp/R-PointHop/target_0.ply.npy")

    pc1 = pc_normalize(pc1)
    pc2 = pc_normalize(pc2)

    low_freq_rec1 = freq_based_sampling(pc1)
    low_freq_rec2 = freq_based_sampling(pc2)

    # min_distances, arg_min_distances = find_distance_pc(pc2, low_freq_rec)
    distances_1 = sklearn.metrics.pairwise.euclidean_distances(pc1, low_freq_rec1)
    distances_2 = sklearn.metrics.pairwise.euclidean_distances(pc2, low_freq_rec2)


    arg_min_1 = np.argmin(distances_1, axis = 1)
    arg_min_2 = np.argmin(distances_2, axis = 1)

    min_1 = np.min(distances_1, axis = 1)
    min_2 = np.min(distances_2, axis = 1)

    sorted_1 = np.argsort(min_1)
    sorted_2 = np.argsort(min_2)

    plot_style_2(pc1, sorted_1[0:32])
    plot_style_2(pc2, sorted_2[0:32])

#visualize_distance_based_sampling()
compare_methods()
exit(0)