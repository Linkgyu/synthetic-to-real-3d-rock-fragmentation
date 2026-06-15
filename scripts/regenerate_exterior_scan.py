from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from io_utils.export_pointcloud import save_ascii_ply, save_labelled_npz
from scanning.occlusion_model import exterior_points_from_viewpoints
from scanning.scan_noise import add_range_noise


SCAN_DIR = ROOT / "data" / "scanned_pointclouds"
TABLE_DIR = ROOT / "outputs" / "tables"

FULL_PATH = SCAN_DIR / "synthetic_rockpile_full_labelled_pointcloud.npz"
OUT_NPZ = SCAN_DIR / "synthetic_rockpile_exterior_only_scan.npz"
OUT_PLY = SCAN_DIR / "synthetic_rockpile_exterior_only_scan.ply"

PARAMS = {
    "angular_resolution_deg": 0.22,
    "height_envelope_grid_m": 0.035,
    "height_envelope_tolerance_m": 0.030,
    "density_keep_fraction": 0.82,
    "range_noise_sigma_m": 0.0015,
    "random_seed": 20260606,
}


def main() -> None:
    full = np.load(FULL_PATH)
    points = full["points_xyz"]
    labels = full["fragment_ids"]
    center = points.mean(axis=0)
    span = points.max(axis=0) - points.min(axis=0)
    radius = float(max(span[0], span[1]) * 1.8 + 0.8)
    z_mid = float(center[2] + 0.35 * span[2])
    z_top = float(points[:, 2].max() + radius)
    viewpoints = np.array(
        [
            [center[0] + radius, center[1], z_mid],
            [center[0] - radius, center[1], z_mid],
            [center[0], center[1] + radius, z_mid],
            [center[0], center[1] - radius, z_mid],
            [center[0], center[1], z_top],
        ],
        dtype=float,
    )

    exterior_points, exterior_labels, exterior_indices = exterior_points_from_viewpoints(
        points,
        labels,
        viewpoint_xyz_list=[vp for vp in viewpoints],
        angular_resolution_deg=PARAMS["angular_resolution_deg"],
        height_envelope_grid_m=PARAMS["height_envelope_grid_m"],
        height_envelope_tolerance_m=PARAMS["height_envelope_tolerance_m"],
    )
    rng = np.random.default_rng(PARAMS["random_seed"])
    n_keep = max(1, int(len(exterior_points) * PARAMS["density_keep_fraction"]))
    keep_indices = rng.choice(len(exterior_points), size=n_keep, replace=False)
    reduced_points = exterior_points[keep_indices]
    reduced_labels = exterior_labels[keep_indices]
    reduced_indices = exterior_indices[keep_indices]
    reduced_points = add_range_noise(
        reduced_points,
        viewpoint_xyz=viewpoints[0],
        sigma_m=PARAMS["range_noise_sigma_m"],
        random_seed=PARAMS["random_seed"] + 1,
    )

    save_labelled_npz(
        OUT_NPZ,
        reduced_points,
        reduced_labels,
        extra={
            "exterior_source_indices": reduced_indices.astype(np.int64),
            "viewpoints_xyz": viewpoints.astype(float),
            "angular_resolution_deg": np.array([PARAMS["angular_resolution_deg"]], dtype=float),
            "height_envelope_grid_m": np.array([PARAMS["height_envelope_grid_m"]], dtype=float),
            "height_envelope_tolerance_m": np.array([PARAMS["height_envelope_tolerance_m"]], dtype=float),
            "scan_filter_version": np.array(["angular_nearest_plus_xy_height_envelope"]),
        },
    )
    save_ascii_ply(OUT_PLY, reduced_points, reduced_labels)

    summary = pd.DataFrame(
        [
            {
                "n_full_points": int(len(points)),
                "n_exterior_points_before_density_reduction": int(len(exterior_points)),
                "n_final_scan_points": int(len(reduced_points)),
                "n_fragments_full_cloud": int(len(np.unique(labels))),
                "n_fragments_visible_after_exterior_filter": int(len(np.unique(exterior_labels))),
                "n_fragments_final_scan": int(len(np.unique(reduced_labels))),
                **PARAMS,
                "scan_filter_version": "angular_nearest_plus_xy_height_envelope",
            }
        ]
    )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_DIR / "exterior_scan_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(OUT_NPZ)


if __name__ == "__main__":
    main()
