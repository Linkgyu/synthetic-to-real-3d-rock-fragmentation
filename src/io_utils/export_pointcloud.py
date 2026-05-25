"""Point-cloud export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_labelled_npz(
    path: Path,
    points_xyz: np.ndarray,
    fragment_ids: np.ndarray,
    extra: dict[str, np.ndarray] | None = None,
) -> None:
    """Save labelled point-cloud arrays to a compressed NPZ file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "points_xyz": np.asarray(points_xyz, dtype=float),
        "fragment_ids": np.asarray(fragment_ids, dtype=np.int64),
    }
    if extra:
        payload.update(extra)
    np.savez_compressed(path, **payload)


def save_ascii_ply(path: Path, points_xyz: np.ndarray, labels: np.ndarray | None = None) -> None:
    """Write a simple ASCII PLY point cloud with optional integer labels.

    This avoids making Open3D mandatory at export time while remaining readable
    by common point-cloud tools.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points_xyz, dtype=float)
    labels_array = None if labels is None else np.asarray(labels, dtype=np.int64)

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if labels_array is not None:
            f.write("property int fragment_id\n")
        f.write("end_header\n")
        if labels_array is None:
            for x, y, z in points:
                f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")
        else:
            for (x, y, z), label in zip(points, labels_array):
                f.write(f"{x:.8f} {y:.8f} {z:.8f} {int(label)}\n")

