"""Virtual LiDAR-style sampling from labelled fragment meshes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass(frozen=True)
class VirtualScan:
    """Labelled virtual scan output."""

    points_xyz: np.ndarray
    fragment_ids: np.ndarray
    ranges_m: np.ndarray
    viewpoint_xyz: np.ndarray


def sample_fragment_surfaces(
    meshes: list[trimesh.Trimesh],
    fragment_ids: list[int] | np.ndarray,
    points_per_m2: float = 15_000.0,
    min_points_per_fragment: int = 80,
    random_seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample labelled points from fragment mesh surfaces.

    Sampling count is proportional to mesh surface area. This is a mesh-based
    stand-in for dense pre-scan surface sampling before viewpoint degradation.
    """

    rng = np.random.default_rng(random_seed)
    all_points = []
    all_labels = []
    for mesh, fragment_id in zip(meshes, fragment_ids):
        n_points = max(min_points_per_fragment, int(mesh.area * points_per_m2))
        points, _ = trimesh.sample.sample_surface(mesh, n_points, seed=rng)
        all_points.append(points)
        all_labels.append(np.full(n_points, int(fragment_id), dtype=np.int64))
    return np.vstack(all_points), np.concatenate(all_labels)


def viewpoint_filter(
    points_xyz: np.ndarray,
    labels: np.ndarray,
    viewpoint_xyz: np.ndarray,
    max_range_m: float | None = None,
    field_of_view_deg: float | None = None,
    look_at_xyz: np.ndarray | None = None,
) -> VirtualScan:
    """Apply simple viewpoint and range filtering to sampled points."""

    points = np.asarray(points_xyz, dtype=float)
    viewpoint = np.asarray(viewpoint_xyz, dtype=float)
    vectors = points - viewpoint
    ranges = np.linalg.norm(vectors, axis=1)
    keep = np.ones(len(points), dtype=bool)

    if max_range_m is not None:
        keep &= ranges <= max_range_m

    if field_of_view_deg is not None and look_at_xyz is not None:
        look_dir = np.asarray(look_at_xyz, dtype=float) - viewpoint
        look_dir = look_dir / np.clip(np.linalg.norm(look_dir), 1e-12, None)
        ray_dir = vectors / np.clip(ranges[:, None], 1e-12, None)
        cos_angle = ray_dir @ look_dir
        keep &= cos_angle >= np.cos(np.deg2rad(field_of_view_deg) / 2.0)

    return VirtualScan(
        points_xyz=points[keep],
        fragment_ids=np.asarray(labels, dtype=np.int64)[keep],
        ranges_m=ranges[keep],
        viewpoint_xyz=viewpoint,
    )

