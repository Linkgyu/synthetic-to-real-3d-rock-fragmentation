"""Open3D point-cloud viewing helpers."""

from __future__ import annotations

import numpy as np


def make_open3d_pointcloud(points_xyz: np.ndarray, colors_rgb: np.ndarray | None = None):
    """Create an Open3D point cloud object.

    The import is local so non-visual scripts can run without opening GUI
    backends until this function is called.
    """

    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points_xyz, dtype=float))
    if colors_rgb is not None:
        cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors_rgb, dtype=float))
    return cloud


def view_pointcloud(points_xyz: np.ndarray, colors_rgb: np.ndarray | None = None) -> None:
    """Open an interactive Open3D viewer for a point cloud."""

    import open3d as o3d

    cloud = make_open3d_pointcloud(points_xyz, colors_rgb)
    o3d.visualization.draw_geometries([cloud])

