"""Scan degradation utilities for virtual point clouds."""

from __future__ import annotations

import numpy as np


def reduce_density(
    points_xyz: np.ndarray,
    labels: np.ndarray,
    keep_fraction: float,
    random_seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly keep a fraction of points."""

    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1].")
    rng = np.random.default_rng(random_seed)
    n_keep = max(1, int(len(points_xyz) * keep_fraction))
    indices = rng.choice(len(points_xyz), size=n_keep, replace=False)
    return points_xyz[indices], labels[indices]


def add_gaussian_noise(
    points_xyz: np.ndarray,
    sigma_m: float,
    random_seed: int = 2026,
) -> np.ndarray:
    """Add isotropic Gaussian coordinate noise."""

    rng = np.random.default_rng(random_seed)
    return np.asarray(points_xyz, dtype=float) + rng.normal(0.0, sigma_m, size=np.asarray(points_xyz).shape)


def add_range_noise(
    points_xyz: np.ndarray,
    viewpoint_xyz: np.ndarray,
    sigma_m: float,
    random_seed: int = 2026,
) -> np.ndarray:
    """Add Gaussian noise along each point's sensor ray."""

    points = np.asarray(points_xyz, dtype=float)
    viewpoint = np.asarray(viewpoint_xyz, dtype=float)
    vectors = points - viewpoint
    ranges = np.linalg.norm(vectors, axis=1)
    directions = vectors / np.clip(ranges[:, None], 1e-12, None)
    rng = np.random.default_rng(random_seed)
    return points + directions * rng.normal(0.0, sigma_m, size=(len(points), 1))

