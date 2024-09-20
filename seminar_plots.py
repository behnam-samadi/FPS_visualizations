from itertools import accumulate

import sklearn
import numpy as np
from requests.packages import target
from sklearn.metrics import pairwise_distances
from torch.fx.experimental.unification.multipledispatch.dispatcher import source

import matplotlib.pyplot as plt
import numpy as np

from frequency_domain_operations import  freq_based_sampling
from utils import read_point_cloud, pc_normalize
from make_registration_data import make_target_frame
from ploting import plot_style_2
from sampling_methods import farthest_point_sample, farthest_point_sample_proposed, random_point_sample
import matplotlib.pyplot as plt
import time

def create_sample():
    source = read_point_cloud(
        "/home/behnam/phd/research/frequency_domain/pythonProject/dataset/raw_frames/airplane_0003.txt")
    source = source[0:1024, 0:3]
    target = make_target_frame(source)
    source_orig = np.copy(source)
    target_orig = np.copy(target)
    source = pc_normalize(source)
    target = pc_normalize(target)
    source_pc_rec = freq_based_sampling(source)
    target_pc_rec = freq_based_sampling(target)

    return source, target, source_pc_rec, target_pc_rec, source_orig, target_orig

def plot2_1():
    source, target, source_pc_rec, target_pc_rec, source_orig, target_orig = create_sample()
    distances_source = sklearn.metrics.pairwise.euclidean_distances(source, source_pc_rec)
    distances_target = sklearn.metrics.pairwise.euclidean_distances(target, target_pc_rec)

    min_distances_source = np.min(distances_source, axis = 1)
    min_distances_target = np.min(distances_target, axis = 1)

    selected_source = np.argsort(min_distances_source)[0:5]
    selected_target = np.argsort(min_distances_target)[0:5]

    selected_target += 1024
    whole = np.concatenate((source_orig, target_orig))
    selecteds = np.concatenate((selected_source, selected_target))
    plot_style_2(whole, selecteds)

def plot2_2():
    source, target, source_pc_rec, target_pc_rec, source_orig, target_orig = create_sample()
    distances_source = sklearn.metrics.pairwise.euclidean_distances(source, source_pc_rec)
    distances_target = sklearn.metrics.pairwise.euclidean_distances(target, target_pc_rec)

def min_pairwise_distance(pc, centroids):
    centroids = centroids.astype(np.int32)
    if centroids.ndim == 2:
        centroids = centroids[0]
    pc_downsmaple = pc[centroids]
    pairwise_distances = sklearn.metrics.pairwise.euclidean_distances(pc_downsmaple, pc_downsmaple)
    pairwise_distances[pairwise_distances==0] = 1e10
    min_dis = np.min(pairwise_distances)
    return min_dis


def plot1_1():
    pc = read_point_cloud(
        "/home/behnam/phd/research/frequency_domain/pythonProject/dataset/raw_frames/airplane_0003.txt")
    pc = pc[:,0:3]
    m_range = range(2, 256, 4)
    fps_orig_distances = []
    rps_distances = []
    fps_proposed_distances = []
    fps_orig_times = []
    rps_times = []
    fps_proposed_times = []
    for npoint in m_range:


        print(npoint)
        fps_orig_time = -time.time()
        fps_orig = farthest_point_sample(pc, npoint)
        fps_orig_time += time.time()
        fps_orig_times.append(fps_orig_time)
        fps_orig_distances.append(min_pairwise_distance(pc, fps_orig))

        rps_time = -time.time()
        rps = random_point_sample(pc, npoint)
        rps_time += time.time()
        rps_times.append(rps_time)
        rps_distances.append(min_pairwise_distance(pc, rps))

        fps_proposed_time = - time.time()
        fps_proposed = farthest_point_sample_proposed(pc, npoint)
        fps_proposed_time += time.time()
        fps_proposed_times.append(fps_proposed_time)
        fps_proposed_distances.append(min_pairwise_distance(pc, fps_proposed))

        print()

    # Accuracy metrics for algorithms 1, 2, and 3
    accuracy_algorithm1 = [0.8, 0.85, 0.9, 0.85, 0.9, 0.85, ...]  # replace with your data
    accuracy_algorithm2 = [0.75, 0.8, 0.85, 0.7, 0.75, 0.8, ...]  # replace with your data
    accuracy_algorithm3 = [0.9, 0.95, 0.95, 0.9, 0.85, 0.9, ...]  # replace with your data

    # Run-time metrics for algorithms 1, 2, and 3
    runtime_algorithm1 = [10, 12, 15, 10, 12, 15, ...]  # replace with your data
    runtime_algorithm2 = [5, 7, 10, 5, 7, 10, ...]  # replace with your data
    runtime_algorithm3 = [20, 25, 30, 20, 25, 30, ...]  # replace with your data
    print(len(fps_proposed_distances))
    m_range = np.arange(1, len(fps_proposed_distances) + 1)
    print("m_range ", m_range)
    # Plot accuracy metrics

    #plt.scatter(m_range, fps_orig_distances)
    plt.plot(m_range, fps_orig_distances)
    #plt.show()

    #plt.plot((m_range, fps_orig_distances), label='original FPS', color='blue')

    #plt.plot((m_range, rps_distances), label='RPS', color='red')
    plt.plot(m_range, rps_distances)
    #plt.plot((m_range, fps_proposed_distances), label='proposed FPS', color='green')
    plt.plot(m_range, fps_proposed_distances)
    plt.show()
    plt.xlabel('Parameter m')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Algorithms')
    plt.legend()
    #plt.show()

    # Plot run-time metrics
    # plt.plot(m_range, runtime_algorithm1, label='Algorithm1', color='blue')
    # plt.plot(m_range, runtime_algorithm2, label='Algorithm2', color='red')
    # plt.plot(m_range, runtime_algorithm3, label='Algorithm3', color='green')
    # plt.xlabel('Parameter m')
    # plt.ylabel('Run-time (seconds)')
    # plt.title('Run-time of Algorithms')
    # plt.legend()
    # plt.show()


def read_accuracy_file(experiment):
    base_address = "/home/behnam/phd/research/results/accuracy_of_fps/"
    file_address = base_address + experiment + ".txt"
    with open(file_address, 'r') as f:
        lines = f.readlines()
    accuracies_list = []
    accuracies = {}
    for i, line in enumerate(lines):
        if "Test Instance Accuracy" in line:
            epoch_line = i-2
            epoch_num = int(lines[epoch_line].split('Epoch')[1].split("(")[0])
            accuracy = float(line.split("Test Instance Accuracy:")[1].split(",")[0])
            accuracies[epoch_num] = accuracy
            accuracies_list.append(accuracy)
    return accuracies_list

def plot3():
    rps_accuracies = read_accuracy_file("rps")
    original_accuracies = read_accuracy_file("original")
    proposed_accuracies = read_accuracy_file("proposed")
    min_len = min(len(rps_accuracies), len(original_accuracies), len(proposed_accuracies))
    rps_accuracies = rps_accuracies[0:min_len]
    original_accuracies = original_accuracies[0:min_len]
    proposed_accuracies = proposed_accuracies[0:min_len]

    epochs = range(1, len(rps_accuracies) + 1)
    plt.plot(epochs, rps_accuracies, label='Random', color='blue', marker='o')
    #plt.plot(epochs, proposed_accuracies, label='Proposed', color='green', marker='s')
    plt.plot(epochs, original_accuracies, label='Original', color='red', marker='^')

    print("rps: ", max(rps_accuracies))
    print("original: ", max(original_accuracies))
    print("proposed: ", max(proposed_accuracies))



    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Accuracies between Methods')

    # Adding a legend to differentiate between the methods
    plt.legend()

    # Show the plot
    plt.show()

    a = 9





    a = 0



