import random
from utils import pc_bounds


def generate_random_float(min_value, max_value):
    return min_value + (max_value - min_value) * random.random()


def add_outlier_to_pc(pc, n_outliers):
    x_min, x_max, y_min, y_max, z_min, z_max = pc_bounds(pc)
    for n in range(n_outliers):
        x = generate_random_float(x_min, x_max)
        y = generate_random_float(y_min, y_max)
        z = generate_random_float(z_min, z_max)
        outlier = np.array([x,y,z])
        outlier = np.expand_dims(outlier, 0)
        pc = np.concatenate((pc, outlier), axis = 0)
    return pc
