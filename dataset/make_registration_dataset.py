import os

import numpy as np

from utils import read_point_cloud
from utils import read_point_cloud, pc_normalize
from make_registration_data import make_target_frame
from outlier import add_outlier_to_pc

def create_dataset():
    frame_size = 1024
    num_outliers = 10
    frames_address = "/home/behnam/phd/research/frequency_domain/pythonProject/dataset/raw_frames/"

    arr = os.listdir(frames_address)

    for file in arr:
        address = frames_address + file
        sample = np.zeros(shape = (frame_size, 4, 3))
        pc = read_point_cloud(address)
        pc = pc_normalize(pc[0:frame_size, 0:3])
        pc2 = make_target_frame(pc)
        pc_ = add_outlier_to_pc(pc, num_outliers)[num_outliers:]
        pc2_ = add_outlier_to_pc(pc2, num_outliers)[num_outliers:]
        sample[:, 0, :] = pc
        sample[:, 1, :] = pc2
        sample[:, 2, :] = pc_
        sample[:, 3, :] = pc2_
        sample_file_name = "/home/behnam/phd/research/frequency_domain/pythonProject/dataset/samples/"+file.split(".txt")[0]
        np.save(sample_file_name, sample)



