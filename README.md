# scientific-computing-proj-1
3D Terrain Estimation using Inverse Distance Weighting (IDW)
We will implement the Inverse Distance Weighting algorithm (slide 1 pg 19) to generate a 3D ocean floor surface, using a sample of sonar timings - probably using "np.random.uniform".
1. Generate a grid of coordinates (NumPy)
2. calculate the hypotenuse distance between grid points and the known sample points,
3. apply the "weighting formula" (from the slides) to estimate the unknown depths.
Our project will compare the speed of a typical nested Python loop against a fully vectorized NumPy approach for generating the surface.
