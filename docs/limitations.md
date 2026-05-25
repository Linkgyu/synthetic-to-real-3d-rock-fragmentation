# Limitations

This repository is a prototype synthetic benchmark.

Important limitations:

- It does not yet contain field LiDAR scans.
- It does not claim site-calibrated fragmentation accuracy.
- It does not implement deep learning segmentation.
- Mesh-derived ground truth may differ from real fractured rock surfaces.
- Convex hull volume estimation is biased for sparse, partial, or concave point clusters.
- DBSCAN is sensitive to point density and parameter choices.
- The occlusion model is an angular-bin approximation, not full optical ray tracing.

These limitations are intentional at the first stage. The project is meant to make error sources visible before moving to real-world validation.

