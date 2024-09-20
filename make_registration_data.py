import numpy as np
# Define the point cloud

def rotation_matrix(axis, angle):
    half_angle = angle / 2
    ca = np.cos(half_angle)
    sa = np.sin(half_angle)
    axis = axis / np.linalg.norm(axis)
    u = axis[0]
    v = axis[1]
    w = axis[2]
    R = np.array([[ca + u ** 2 * (1 - ca), u * v * (1 - ca) - w * sa, u * w * (1 - ca) + v * sa],
                  [u * v * (1 - ca) + w * sa, ca + v ** 2 * (1 - ca), v * w * (1 - ca) - u * sa],
                  [u * w * (1 - ca) - v * sa, v * w * (1 - ca) + u * sa, ca + w ** 2 * (1 - ca)]])
    return R

def make_target_frame(source_frame):
    # Define the bounds for rotation and translation
    rotation_bounds = [(0, 45), (0, 45), (0, 45)]  # 0° to 45°
    translation_bounds = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]  # -0.5 to 0.5
    translation_bounds = np.array(translation_bounds)
    translation_bounds *= 2
    rotation_bounds = np.array(rotation_bounds)
    rotation_bounds *= 2

    # Apply random rotation to the point cloud
    for i in range(3):
        rotation_angle = np.random.uniform(*rotation_bounds[i])
        rotation_angle = 90
        axis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]][i])  # Select the current axis
        source_frame = np.dot(source_frame - np.mean(source_frame, axis=0), rotation_matrix(axis, np.radians(rotation_angle))) + np.mean(source_frame, axis=0)

    # Apply random translation to the point cloud
    for i in range(3):
        translation_vector = np.array([0, 0, 0]).astype(np.float64)
        rotation_bounds *= 2
        translation_vector[i] = np.random.uniform(translation_bounds[i][0], translation_bounds[i][1])
        print(translation_vector)
        source_frame += translation_vector
        #translation_vector = np.random.uniform(translation_bounds[i][0], translation_bounds[i][1])

    #translation_vector = np.random.uniform(*translation_bounds).reshape(1, 3)
    #source_frame += np.random.uniform(*translation_bounds).reshape(1, 3)

    # Define the rotation matrix function
    noise = np.random.normal(0, 0.01, size=(1024, 3))
    source_frame += noise
    return source_frame