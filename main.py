"""
Team
    Takuro Kawamura
    Zoe Rochelle

September 2024
Scientific Computing: Project 1
Dr. J

Topic
    3D Terrain Estimation using Inverse Distance Weighting (IDW)

Background
    The goal of this proj is to compare the speed of a typical nested Python loop against a fully vectorized NumPy approach for generating a 3D ocean floor surface using the Inverse Distance Weighting (IDW) algorithm.
"""


import numpy as np
import time
# import sys

def generate_sample_points(num_points: int) -> np.ndarray:
    """
    Returns: an array of random 3D points between (0, 0, 0) and (1, 1, 1).

        [
            (X, Y, Z),
            (X, Y, Z),
            ...
        ]

    """

    rng = np.random.default_rng()
    points_3d = rng.random((num_points, 3))
    # print("points:\n", points_3d)

    return points_3d


def construct_grid(dimensions: tuple) -> tuple:     # returns a matrix
    """
    Construct a grid of coordinates.

    Returns: a tuple of 2D arrays representing the X and Y coordinates of the grid.

        (
            [ 0,  1,  2, ..., 97, 98, 99],
            [ 0,  1,  2, ..., 97, 98, 99],
            ...
        )

    """


    # X Matrix = horizontal
    x = np.linspace(0, dimensions[0] - 1, dimensions[0], dtype=int)     # 0, to 99, 100 ints

    # Y Matrix = vertical
    y = np.linspace(0, dimensions[1] - 1, dimensions[1], dtype=int)

    # meshgrid() takes two 1D arrays of coordinates and expands them into a pair of 2D coordinate matrices.
    return np.meshgrid(x, y)


def calc_idw_height(xi, yi, p, num_samples, grid_x, grid_y, samples_x, samples_y,  samples_z):
    """
    Original for-loop implementation - taken from the slide 19

    Use Inverse Distance Weighting (IDW), to calculate the
    estimated depth at a single grid coordinate using a for-loop.

    Parameters:
    - xi (int): Column index on the target grid.
    - yi (int): Row index on the target grid.

    - p (float): Power parameter for distance weighting.

    - num_samples (int): Total number of known sample points.
    - grid_x, grid_y (ndarray): 2D coordinate matrices for the target grid.

    - samples_x, samples_y (ndarray): 1D arrays of sample spatial coordinates.
    - samples_z (ndarray): 1D array of known sample depths.

    Returns:
    - float: the IDW height (the estimated depth at the specified grid coordinate)

    """

    sum_weight = 0.0
    sum_height_weight = 0.0
    for si in range(num_samples):
        distance = np.hypot(
                grid_x[yi, xi] - samples_x[si],
                grid_y[yi, xi] - samples_y[si]
            )
        if distance == 0:
            return samples_z[si]
        weight = 1 / np.power(distance, p)
        sum_weight += weight
        sum_height_weight += samples_z[si] * weight
    return sum_height_weight / sum_weight


def vectorized_calc_idw_height(p, grid_x, grid_y, samples_x, samples_y, samples_z):
    """
    Vectorized numpy broadcasting implementation.

    Using numpy's broadcasting capabilities, apply the distance and weighting calculations
    across the entire grid simultaneously to eliminate the slower loop method.

    source: https://numpy.org/doc/stable/user/basics.broadcasting.html
    """

    # 1. Stretch the 1D arrays (samples_x and samples_y) into 3D arrays so they can be broadcasted against the 2D grid matrices.
    samples_broadcast_x = samples_x.reshape(-1, 1, 1)   # (-1, 1, 1) creates 2 new dimensions for broadcasting
    samples_broadcast_y = samples_y.reshape(-1, 1, 1)

    # 2. Apply 'np.hypot' against the broadcasted coordinate arrays
    # this makes numpy calculate the coordinate differences across every grid point simultaneously.

    # distance between two points: sqrt( (x_2 - x_1)^2 + (y_2 - y_1)^2 )
    distance = np.hypot(
        grid_x - samples_broadcast_x,
        grid_y - samples_broadcast_y
    )

    # 3. Apply the weighting formula to the new distance matrix
    # this will calculate the weights for each sample point relative to every grid point.
    # numpy will automatically apply the power parameter to every element.
    weight = 1 / np.power(distance, p)

    # 4. Compute the final IDW estimated terrain grid by broadcasting Z-coordinates,
    samples_broadcast_z = samples_z.reshape(-1, 1, 1)

    # multiply by weights, sum across all sample points
    numerator = np.sum(weight * samples_broadcast_z, axis=0)
    denominator = np.sum(weight, axis=0)

    # then divide the weighted height sum by the total weight sum
    final_estimated_depth_matrix = numerator / denominator

    return final_estimated_depth_matrix


if __name__ == "__main__":

    # 1. generate 220 sample points containing X, Y, and Z coordinates to represent the known sonar timings.
    sample_points_3d = generate_sample_points(220)
    # print(sample_points_3d)


    # 2. construct a grid of coordinates (X, Y) to represent the unknown depths.
    X, Y = construct_grid((100, 100)) # change this to (1000, 1000) after testing to see how difference in perf.


    # 3. Extract individual X, Y, and Z coordinate arrays from the 3D sample points.
    # we need to separate the 3D coordinates into individual variables to use in the formula
    samples_x = sample_points_3d[:, 0]
    samples_y = sample_points_3d[:, 1]  # slicing the 2D array to get every row in the second column
    samples_z = sample_points_3d[:, 2]




    # 5. benchmark the 2 approaches
    start_time = time.perf_counter()

    ## Benchmark the for loop across the entire 100x100 grid
    rows, cols = X.shape    # get the dimensions of the grid - shape returns a tuple of (rows, cols)
    baseline_grid = np.zeros((rows, cols))  # creates a grid of zeros to store the estimated depths
    for r in range(rows):
        for c in range(cols):
            baseline_grid[r, c] = calc_idw_height(r, c, 2, 220, X, Y, samples_x, samples_y, samples_z)

    end_time = time.perf_counter()
    print(f"For-loop approach took {end_time - start_time} seconds")

    ## Benchmark the Vectorized approach
    start_time = time.perf_counter()
    test_depth_2 = vectorized_calc_idw_height(2, X, Y, samples_x, samples_y, samples_z)
    end_time = time.perf_counter()
    print(f"Vectorized approach took {end_time - start_time} seconds")



## output ##
# For-loop approach took 4.379528033001407 seconds
# Vectorized approach took 0.05964729400147917 seconds
