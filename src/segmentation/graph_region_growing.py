"""Graph-based surface segmentation for exterior rockpile point clouds."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _relabel_nonnegative(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    out = np.full(len(labels), -1, dtype=int)
    valid = labels >= 0
    if np.any(valid):
        _, inv = np.unique(labels[valid], return_inverse=True)
        out[valid] = inv
    return out


def graph_region_growing_segmentation(
    points_xyz: np.ndarray,
    normals: np.ndarray,
    curvature: np.ndarray,
    radius_m: float = 0.040,
    max_normal_angle_deg: float = 38.0,
    max_curvature_delta: float = 0.055,
    max_vertical_jump_m: float = 0.055,
    min_cluster_points: int = 35,
) -> np.ndarray:
    """Segment a scan by connected components in a surface-compatibility graph.

    Two points are connected only when they are spatial neighbours and their
    local surface geometry is compatible. This is deliberately still classical
    and lightweight; it is meant as a stronger baseline than XYZ-only DBSCAN,
    not as a solved fragment-recognition algorithm.
    """

    points = np.asarray(points_xyz, dtype=float)
    normals = np.asarray(normals, dtype=float)
    curvature = np.asarray(curvature, dtype=float)
    n_points = len(points)
    if not (len(normals) == len(curvature) == n_points):
        raise ValueError("points_xyz, normals, and curvature must have the same length")
    if n_points == 0:
        return np.empty(0, dtype=int)

    tree = cKDTree(points)
    pairs = np.array(list(tree.query_pairs(radius_m)), dtype=np.int64)
    parent = np.arange(n_points, dtype=np.int64)
    rank = np.zeros(n_points, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    if len(pairs):
        i = pairs[:, 0]
        j = pairs[:, 1]
        normal_dot = np.abs(np.sum(normals[i] * normals[j], axis=1))
        normal_ok = normal_dot >= np.cos(np.deg2rad(max_normal_angle_deg))
        curvature_ok = np.abs(curvature[i] - curvature[j]) <= max_curvature_delta
        vertical_ok = np.abs(points[i, 2] - points[j, 2]) <= max_vertical_jump_m
        keep = normal_ok & curvature_ok & vertical_ok
        for a, b in pairs[keep]:
            union(int(a), int(b))

    roots = np.array([find(i) for i in range(n_points)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)

    counts = np.bincount(labels)
    small = counts[labels] < int(min_cluster_points)
    labels = labels.astype(int)
    labels[small] = -1
    return _relabel_nonnegative(labels)


def cluster_size_table(labels: np.ndarray) -> dict[str, float]:
    """Return simple cluster-count diagnostics."""

    labels = np.asarray(labels)
    valid = labels >= 0
    counts = np.bincount(labels[valid]) if np.any(valid) else np.array([], dtype=int)
    return {
        "n_clusters": int(len(counts)),
        "noise_fraction": float(np.mean(~valid)) if len(labels) else 0.0,
        "median_cluster_points": float(np.median(counts)) if len(counts) else 0.0,
        "max_cluster_points": int(counts.max()) if len(counts) else 0,
    }
