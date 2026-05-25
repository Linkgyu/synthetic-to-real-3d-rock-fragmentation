# Method Overview

This project defines a synthetic benchmark for 3D rock fragmentation analysis from point clouds.

The benchmark separates the problem into controlled stages:

1. Reference rockpile generation or loading
2. Ground-truth fragment ID and volume assignment
3. Virtual LiDAR-style point sampling
4. Scan degradation
5. Classical point-cloud segmentation
6. Fragment size and PSD estimation
7. Ground-truth comparison
8. Sensitivity analysis for P80 error

The first implementation uses classical algorithms only. DBSCAN is the baseline segmentation method, convex hull volume is the first size-estimation baseline, and P10/P50/P80 are computed from volume-weighted cumulative PSD curves.

The purpose is not to replace field measurements. The purpose is to create a controlled environment where the effect of resolution, noise, and occlusion can be measured against known synthetic ground truth.

