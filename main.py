## scientific-computing-proj-1

## topic: 3D Terrain Estimation using Inverse Distance Weighting (IDW)

## the goal of this proj is to compare the speed of a typical nested Python loop against a fully vectorized NumPy approach for generating a 3D ocean floor surface using Inverse Distance Weighting (IDW) algorithm.

# Implement the Inverse Distance Weighting algorithm (slide 1 pg 19) to generate a 3D ocean floor surface, using a sample of sonar timings - probably using "np.random.uniform".
# 1. Generate a grid of coordinates (NumPy)
# 2. calculate the hypotenuse distance between grid points and the known sample points
# 3. apply the "weighting formula" (from the slides) to estimate the unknown depths.
# Our project will compare the speed of a typical nested Python loop against a fully vectorized NumPy approach for generating the surface.

import numpy as np
import sys


def generate_sample_points(num_points):
    """
    Generate a set of random 3D points.
    """
    rng = np.random.default_rng()
    points_3d = rng.random((num_points, 3))
    # print("\n3D Points:\n", points_3d)
    return points_3d


def construct_grid(dimensions):
    """
    Construct a grid of coordinates.
    """
    x = np.linspace(0, dimensions[0] - 1, dimensions[0], dtype=int)
    y = np.linspace(0, dimensions[1] - 1, dimensions[1], dtype=int)
    X, Y = np.meshgrid(x, y)

    # X Matrix tells you how far right to go - horizontally
    print("X-coordinates:\n", X)

    # Y Matrix tells you how far up to go - vertically
    print("\nY-coordinates:\n", Y)

    return X, Y


def calc_idw_height(xi, yi, p, num_samples, grid_x, grid_y, samples_x, samples_y,  samples_z):
    """
    baseline for-loop implementation
    taken from the slide 19
    - will need to adjust the args
    """
    sum_weight = 0.0
    sum_height_weight = 0.0
    for si in range(num_samples):
        distance = np.hypot(
                grid_x[xi, xi] - samples_x[si],
                grid_y[yi, yi] - samples_y[si]
            )
        if distance == 0:
            return samples_z[si]
        weight = 1 / np.power(distance, p)
        sum_weight += weight
        sum_height_weight += samples_z[si] * weight
    return sum_height_weight / sum_weight


# 1. generate 220 sample points containing X, Y, and Z coordinates to represent the known sonar timings.
sample_points = generate_sample_points(220)

# 2. construct a grid of coordinates (X, Y) to represent the unknown depths.
X, Y = construct_grid((100, 100)) # change this to (1000, 1000) after testing to get a more high resolution map.

# 3. re-create the test-case, "baseline" for-loop from the slide to compare against our vectorized approach.
test_depth = calc_idw_height(0, 0, 2)


sys.exit()
