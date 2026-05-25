"""Fragment volume estimates from segmented point clouds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import trimesh


def estimate_cluster_volumes_convex_hull(points_xyz: np.ndarray, cluster_labels: np.ndarray) -> pd.DataFrame:
    """Estimate cluster volumes using convex hulls.

    Convex hull volume is a simple baseline. It tends to overestimate concave
    fragments and becomes unstable for sparse or heavily occluded clusters.
    """

    points = np.asarray(points_xyz, dtype=float)
    labels = np.asarray(cluster_labels)
    rows = []
    for label in sorted(set(labels)):
        if label < 0:
            continue
        cluster = points[labels == label]
        if len(cluster) < 4:
            volume = 0.0
        else:
            try:
                volume = abs(float(trimesh.points.PointCloud(cluster).convex_hull.volume))
            except Exception:
                volume = 0.0
        rows.append({"cluster_id": int(label), "n_points": int(len(cluster)), "estimated_volume_m3": volume})
    return pd.DataFrame(rows)

