"""Surface-normal and curvature features for exterior rockpile scans."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def estimate_normals_curvature(points_xyz: np.ndarray, k_neighbors: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Estimate local PCA normals and curvature for each point.

    Curvature is defined as the smallest local covariance eigenvalue divided by
    the sum of eigenvalues. It is not a physical fracture metric; it is a local
    geometric roughness cue for classical point-cloud segmentation.
    """

    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")
    if len(points) < 4:
        return np.zeros_like(points), np.zeros(len(points), dtype=float)

    k = int(np.clip(k_neighbors, 4, max(4, len(points) - 1)))
    tree = cKDTree(points)
    _, neighbour_idx = tree.query(points, k=k + 1, workers=-1)
    neighbour_idx = neighbour_idx[:, 1:]

    neighbourhoods = points[neighbour_idx]
    centered = neighbourhoods - neighbourhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / max(k - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    normals = eigenvectors[:, :, 0]
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)

    # Make the sign deterministic. The segmentation uses abs(dot), but stable
    # orientation makes visual checks and saved features easier to inspect.
    flip = normals[:, 2] < 0
    normals[flip] *= -1.0

    curvature = eigenvalues[:, 0] / np.clip(eigenvalues.sum(axis=1), 1e-12, None)
    return normals.astype(np.float32), curvature.astype(np.float32)


def normal_angle_degrees(normals_a: np.ndarray, normals_b: np.ndarray) -> np.ndarray:
    """Return unsigned normal-angle differences in degrees."""

    dots = np.abs(np.sum(normals_a * normals_b, axis=-1))
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(np.arccos(dots))
