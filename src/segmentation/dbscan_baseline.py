"""DBSCAN baseline segmentation for rock fragment point clouds."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN


def segment_dbscan(
    points_xyz: np.ndarray,
    eps_m: float = 0.04,
    min_samples: int = 12,
) -> np.ndarray:
    """Segment a point cloud using DBSCAN.

    DBSCAN label `-1` indicates noise. This baseline is intentionally simple
    and transparent for early synthetic-to-real benchmarking.
    """

    points = np.asarray(points_xyz, dtype=float)
    model = DBSCAN(eps=eps_m, min_samples=min_samples)
    return model.fit_predict(points)

