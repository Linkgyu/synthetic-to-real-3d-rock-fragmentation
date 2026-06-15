"""Simple angular-bin occlusion model for virtual scans."""

from __future__ import annotations

import numpy as np


def keep_nearest_by_angular_bin(
    points_xyz: np.ndarray,
    labels: np.ndarray,
    viewpoint_xyz: np.ndarray,
    angular_resolution_deg: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the nearest point in each azimuth/elevation bin.

    This approximates line-of-sight occlusion for a single static scan position.
    It is not a full ray tracer, but it captures the main benchmark effect:
    points behind nearer rocks are removed.
    """

    points = np.asarray(points_xyz, dtype=float)
    viewpoint = np.asarray(viewpoint_xyz, dtype=float)
    vectors = points - viewpoint
    ranges = np.linalg.norm(vectors, axis=1)
    directions = vectors / np.clip(ranges[:, None], 1e-12, None)

    azimuth = np.rad2deg(np.arctan2(directions[:, 1], directions[:, 0]))
    elevation = np.rad2deg(np.arcsin(np.clip(directions[:, 2], -1.0, 1.0)))
    az_bin = np.floor(azimuth / angular_resolution_deg).astype(int)
    el_bin = np.floor(elevation / angular_resolution_deg).astype(int)

    nearest: dict[tuple[int, int], tuple[float, int]] = {}
    for idx, key in enumerate(zip(az_bin, el_bin)):
        current = nearest.get(key)
        if current is None or ranges[idx] < current[0]:
            nearest[key] = (float(ranges[idx]), idx)

    keep_indices = np.array([idx for _, idx in nearest.values()], dtype=int)
    keep_indices.sort()
    return points[keep_indices], np.asarray(labels, dtype=np.int64)[keep_indices]


def keep_xy_height_envelope(
    points_xyz: np.ndarray,
    labels: np.ndarray,
    source_indices: np.ndarray | None = None,
    grid_resolution_m: float = 0.035,
    z_tolerance_m: float = 0.030,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only points close to the upper surface envelope in each XY cell.

    The angular-bin model removes many line-of-sight occlusions, but a point-only
    ray approximation can still retain samples from interior contact surfaces.
    A photogrammetric rockpile surface is better represented by the upper
    height envelope over the pile footprint. This filter keeps only points whose
    z-coordinate is within ``z_tolerance_m`` of the maximum z in their XY grid
    cell and returns the corresponding original source indices.
    """

    points = np.asarray(points_xyz, dtype=float)
    labels_array = np.asarray(labels, dtype=np.int64)
    if source_indices is None:
        source_array = np.arange(len(points), dtype=np.int64)
    else:
        source_array = np.asarray(source_indices, dtype=np.int64)

    if len(points) == 0:
        return points, labels_array, source_array

    origin_xy = points[:, :2].min(axis=0)
    cells = np.floor((points[:, :2] - origin_xy) / float(grid_resolution_m)).astype(np.int64)
    keys = cells[:, 0] * 10_000_000 + cells[:, 1]
    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]

    keep = np.zeros(len(points), dtype=bool)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_keys[end] == sorted_keys[start]:
            end += 1
        idx = order[start:end]
        z_max = float(points[idx, 2].max())
        keep[idx] = points[idx, 2] >= z_max - float(z_tolerance_m)
        start = end

    kept = np.flatnonzero(keep)
    return points[kept], labels_array[kept], source_array[kept]


def exterior_points_from_viewpoints(
    points_xyz: np.ndarray,
    labels: np.ndarray,
    viewpoint_xyz_list: list[np.ndarray],
    angular_resolution_deg: float = 0.20,
    height_envelope_grid_m: float | None = 0.035,
    height_envelope_tolerance_m: float = 0.030,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep points visible from at least one exterior viewpoint.

    Each viewpoint keeps the nearest point per azimuth/elevation bin. The union
    approximates an outside-only LiDAR survey and removes many internal or
    mutually occluded surfaces from a fully labelled synthetic point cloud.

    Returns points, labels, and original point indices.
    """

    points = np.asarray(points_xyz, dtype=float)
    labels_array = np.asarray(labels, dtype=np.int64)
    kept_indices: list[np.ndarray] = []

    for viewpoint in viewpoint_xyz_list:
        viewpoint = np.asarray(viewpoint, dtype=float)
        vectors = points - viewpoint
        ranges = np.linalg.norm(vectors, axis=1)
        directions = vectors / np.clip(ranges[:, None], 1e-12, None)
        azimuth = np.rad2deg(np.arctan2(directions[:, 1], directions[:, 0]))
        elevation = np.rad2deg(np.arcsin(np.clip(directions[:, 2], -1.0, 1.0)))
        az_bin = np.floor(azimuth / angular_resolution_deg).astype(int)
        el_bin = np.floor(elevation / angular_resolution_deg).astype(int)

        nearest: dict[tuple[int, int], tuple[float, int]] = {}
        for idx, key in enumerate(zip(az_bin, el_bin)):
            current = nearest.get(key)
            if current is None or ranges[idx] < current[0]:
                nearest[key] = (float(ranges[idx]), idx)
        kept_indices.append(np.fromiter((idx for _, idx in nearest.values()), dtype=int))

    union_indices = np.unique(np.concatenate(kept_indices))
    exterior_points = points[union_indices]
    exterior_labels = labels_array[union_indices]
    if height_envelope_grid_m is not None:
        return keep_xy_height_envelope(
            exterior_points,
            exterior_labels,
            source_indices=union_indices,
            grid_resolution_m=height_envelope_grid_m,
            z_tolerance_m=height_envelope_tolerance_m,
        )
    return exterior_points, exterior_labels, union_indices
