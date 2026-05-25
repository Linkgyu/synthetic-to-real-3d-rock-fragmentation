"""Lightweight region-growing placeholder for classical segmentation."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def radius_connected_components(points_xyz: np.ndarray, radius_m: float) -> np.ndarray:
    """Return connected components under a fixed neighbour radius.

    This is a minimal region-growing baseline. It is useful for comparisons
    with DBSCAN because it has no density threshold and therefore over-connects
    more easily under occlusion.
    """

    points = np.asarray(points_xyz, dtype=float)
    tree = cKDTree(points)
    pairs = tree.query_pairs(radius_m)
    parent = np.arange(len(points))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i, j in pairs:
        union(i, j)

    roots = np.array([find(i) for i in range(len(points))])
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(int)

