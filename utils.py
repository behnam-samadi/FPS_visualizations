import numpy as np
from math import floor


def pc_bounds(pc):
    return min(pc[:,0]), max(pc[:,0]), min(pc[:,1]), max(pc[:,1]), min(pc[:,2]), max(pc[:,2])
def euclid_dis(point1, point2):
    #print("---------------euc:::")
    #print(point1.shape)
    #print(point2.shape)
    #print(point1)
    #print(point2)
    #print(type(point1))
    #print(type(point2))
    #print("---------------")
    #print( np.sum(point1 - point2)**2)
    result = np.sqrt(np.sum(((point1 - point2)**2), axis=0))
    #result = np.abs(np.sum(point1) - np.sum(point2))
    #print("result:")
    #print(result)
    return result

def make_point_cloud_from_centroids(points, centroids):
    pass

def read_point_cloud(address):
    with open(address, 'r') as f:
        lines = f.readlines()
    points = []
    for line in lines:
        values = line.split('\n')[0].split(',')
        point = [float(v) for v in values]
        points.append(point)
    return np.array(points)


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def calculate_min_distance(points, centroids):
    # print("first:")
    # print(points.shape)
    # print(centroids.shape)
    # print("--------------")
    #points = points[0]
    #centroids = centroids[0]
    pairwise_distances = np.zeros(shape=(centroids.shape[0], centroids.shape[0]))
    # print(pairwise_distances.shape)
    for i in range(centroids.shape[0]):
        for j in range(centroids.shape[0]):
            if (i == j):
                pairwise_distances[i][j] = 1e10
            else:
                centroid1 = int(centroids[i])
                centroid2 = int(centroids[j])
                pairwise_distances[i][j] = euclid_dis(points[centroid1], points[centroid2])
    mins = np.min(pairwise_distances, 1)
    # print("mins: ", mins, mins.shape)

    print("pairwise_distances: ", np.mean(mins))
    return np.mean(mins)


def calculate_pc_min_distance(points):
    # print("first:")
    # print(points.shape)
    # print(centroids.shape)
    # print("--------------")
    #points = points[0]
    pairwise_distances = np.zeros(shape=(points.shape[0], points.shape[0]))
    # print(pairwise_distances.shape)
    for i in range(points.shape[0]):
        for j in range(points.shape[0]):
            if (i == j):
                #print(i)
                pairwise_distances[i][j] = 1e10
            else:
                pairwise_distances[i][j] = euclid_dis(points[i], points[j])
    mins = np.min(pairwise_distances, 1)
    # print("mins: ", mins, mins.shape)

    print("pairwise_distances: ", np.mean(mins))
    return np.mean(mins)


def count_occupied_voxels(points):

    # assuming your point cloud is stored in a numpy array `points` with shape (n, 3)
    # set the grid resolution
    num_voxel_x = 1000
    x_min, x_max, y_min, y_max, z_min, z_max = pc_bounds(points)

    x_voxel_size = (x_max - x_min)/num_voxel_x
    y_voxel_size = x_voxel_size
    z_voxel_size = x_voxel_size

    # calculate the number of voxels in each direction
    nx = int((x_max - x_min) / x_voxel_size) + 1
    ny = int((y_max - y_min) / y_voxel_size) + 1
    nz = int((z_max - z_min) / z_voxel_size) + 1

    # create a 3D array to store the voxel occupancy
    grid = np.zeros((nx, ny, nz), dtype=int)

    # iterate over the points and assign occupancy to the corresponding voxels
    for point in points:
        x_idx = floor((point[0]- x_min) / x_voxel_size)
        y_idx = floor((point[1] - y_min)/ y_voxel_size)
        z_idx = floor((point[2] - z_min)/ z_voxel_size)
        grid[x_idx, y_idx, z_idx] += 1

    # count the number of occupied voxels
    occupied_voxels = np.sum(grid > 0)
    return occupied_voxels


def make_full_pc(pc, num_voxel_x = 50):
    x_min, x_max, y_min, y_max, z_min, z_max = pc_bounds(pc)
    x_voxel_size = ((x_max - x_min) / num_voxel_x)
    y_voxel_size = x_voxel_size
    z_voxel_size = x_voxel_size
    output = []

    nx = int((x_max - x_min) / x_voxel_size) + 1
    ny = int((y_max - y_min) / y_voxel_size) + 1
    nz = int((z_max - z_min) / z_voxel_size) + 1

    for x_step in range(nx):
        for y_step in range(ny):
            for z_step in range(nz):
                print(x_step, y_step, z_step)
                output.append([x_min + x_step, y_min + y_step, z_min + z_step])
    output = np.array(output)
    return output


