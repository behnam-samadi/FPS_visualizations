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
from seminar_plots import *
import h5py
from Registration.RPointHop.modelnet40 import data_load
from make_registration_data import make_target_frame
import os
from dataset.make_registration_dataset import create_dataset
from pure_proposed_registration import dis_to_rec_based_register



#from various_tests import result


def compare_methods():
    frames_address = "dataset/samples/"
    arr = os.listdir(frames_address)
    for file in arr:
        print(file)
        address = frames_address + file
        sample = np.load(address)
        pc1 = sample[:, 0, :]
        pc2 = sample[:, 1, :]
        proposed_time = - time.time()
        result_proposed = dis_to_rec_based_register(pc1, pc2)
        proposed_time += time.time()

        baseline_time = -time.time()
        result_rpointhop = register(pc1, pc2)
        baseline_time += time.time()

        input = np.concatenate((pc1, pc2))
        output1 = np.concatenate((result_rpointhop, pc2))
        output2 = np.concatenate((result_proposed, pc2))
        plot_style_2(input)
        plot_style_2(output1)
        plot_style_2(output2)


        print(baseline_time, proposed_time)

        print(rmse(result_proposed, pc2))
        print(rmse(result_rpointhop, pc2))


        # whole2 = np.concatenate((result2, pc2))
        # plot_style_2(whole2)
        # result1 = register(pc, pc2)
        # whole1 = np.concatenate((result1, pc2))
        # plot_style_2(whole1)
        #
        #
        #
        #
        #
        # exit(0)
        # result4 = register_orig(pc_, pc2_)
        # result3 = register(pc_, pc2_)
        #
        #
        # distances = sklearn.metrics.pairwise.euclidean_distances(pc2, result1)
        # score = np.sum(np.min(distances ** 2, axis=0) ** 2)
        # print(rmse(result1, pc2))
        # #print("score: ", score)
        #
        # distances = sklearn.metrics.pairwise.euclidean_distances(pc2, result2)
        # score = np.sum(np.min(distances ** 2, axis=0) ** 2)
        # print(rmse(result2, pc2))
        # #print("score: ", score)
        #
        # distances = sklearn.metrics.pairwise.euclidean_distances(pc2_, result3)
        # score = np.sum(np.min(distances ** 2, axis=0) ** 2)
        # print(rmse(result3, pc2_))
        # #print("score: ", score)
        #
        # distances = sklearn.metrics.pairwise.euclidean_distances(pc2_, result4)
        # score = np.sum(np.min(distances, axis=0) ** 2)
        # print(rmse(result4, pc2_))
        # #print("score: ", score)



def visualize_distance_based_sampling():
    #Tests on visualizing distance aware sampling
    sample = np.load("/home/behnam/phd/research/frequency_domain/pythonProject/dataset/samples/monitor_0011.npy")
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

    fps1 = farthest_point_sample(pc1, 128)
    fps2 = farthest_point_sample(pc2, 128)
    plot_style_2(pc1, sorted_1[::8])
    plot_style_2(pc2, sorted_2[::8])


def compare_and_visualize_registration():
    sample = np.load("dataset/samples/monitor_0004.npy")
    pc1 = sample[:,0,:]
    pc2 = sample[:, 1, :]





    pc1_ = sample[:, 2, :]
    pc2_ = sample[:, 3, :]


    result1 = register(pc1, pc2)
    print(rmse(result1, pc2))
    exit(0)

    result2 = register_orig(pc1, pc2)
    result3 = register(pc1_, pc2_)
    result4 = register_orig(pc1_, pc2_)

    whole1 = np.concatenate((pc2, result1))
    whole2 = np.concatenate((pc2, result2))
    whole3 = np.concatenate((pc2_, result3))
    whole4 = np.concatenate((pc2_, result4))

    plot_style_2(whole1)
    plot_style_2(whole2)
    plot_style_2(whole3)
    plot_style_2(whole4)

    plot_style_2(whole1, list(range(1024, 2048)))
    plot_style_2(whole2, list(range(1024, 2048)))
    plot_style_2(whole3, list(range(1024, 2048)))
    plot_style_2(whole4, list(range(1024, 2048)))

    distances = sklearn.metrics.pairwise.euclidean_distances(pc2, result1)
    score = np.sum(np.min(distances ** 2, axis=0) ** 2)
    print(rmse(result1, pc2))
    # print("score: ", score)

    distances = sklearn.metrics.pairwise.euclidean_distances(pc2, result2)
    score = np.sum(np.min(distances ** 2, axis=0) ** 2)
    print(rmse(result2, pc2))
    # print("score: ", score)

    distances = sklearn.metrics.pairwise.euclidean_distances(pc2_, result3)
    score = np.sum(np.min(distances ** 2, axis=0) ** 2)
    print(rmse(result3, pc2_))
    # print("score: ", score)

    distances = sklearn.metrics.pairwise.euclidean_distances(pc2_, result4)
    score = np.sum(np.min(distances, axis=0) ** 2)
    print(rmse(result4, pc2_))
    # print("score: ", score)
    a = 9




plot3()
#plot1_1()
#visualize_distance_based_sampling()
#compare_methods()
#create_dataset()
#compare_and_visualize_registration()
exit(0)