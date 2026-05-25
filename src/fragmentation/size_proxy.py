"""Size proxy estimators for segmented point-cloud clusters."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_bounding_box_sizes(points_xyz: np.ndarray, cluster_labels: np.ndarray) -> pd.DataFrame:
    """Estimate cluster size using axis-aligned bounding box diagonal."""

    points = np.asarray(points_xyz, dtype=float)
    labels = np.asarray(cluster_labels)
    rows = []
    for label in sorted(set(labels)):
        if label < 0:
            continue
        cluster = points[labels == label]
        if len(cluster) == 0:
            continue
        span = cluster.max(axis=0) - cluster.min(axis=0)
        rows.append(
            {
                "cluster_id": int(label),
                "n_points": int(len(cluster)),
                "bbox_x_m": float(span[0]),
                "bbox_y_m": float(span[1]),
                "bbox_z_m": float(span[2]),
                "bbox_diagonal_m": float(np.linalg.norm(span)),
            }
        )
    return pd.DataFrame(rows)

