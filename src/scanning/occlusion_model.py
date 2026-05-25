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


def exterior_points_from_viewpoints(
    points_xyz: np.ndarray,
    labels: np.ndarray,
    viewpoint_xyz_list: list[np.ndarray],
    angular_resolution_deg: float = 0.20,
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
    return points[union_indices], labels_array[union_indices], union_indices
