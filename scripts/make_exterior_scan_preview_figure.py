from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FULL_PATH = ROOT / "data" / "scanned_pointclouds" / "synthetic_rockpile_full_labelled_pointcloud.npz"
SCAN_PATH = ROOT / "data" / "scanned_pointclouds" / "synthetic_rockpile_exterior_only_scan.npz"
OUT_PATH = ROOT / "outputs" / "figures" / "synthetic_rockpile_exterior_scan_preview.png"
SLICE_HALF_WIDTH_M = 0.08
AXIS_HALF_WIDTH_M = 1.25
Z_AXIS_MAX_M = 1.25


def style_axes(ax) -> None:
    ax.set_xlim(-AXIS_HALF_WIDTH_M, AXIS_HALF_WIDTH_M)
    ax.set_ylim(-AXIS_HALF_WIDTH_M, AXIS_HALF_WIDTH_M)
    ax.set_zlim(0.0, Z_AXIS_MAX_M)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_zticks([0.0, 0.5, 1.0])
    ax.set_box_aspect((2.0 * AXIS_HALF_WIDTH_M, 2.0 * AXIS_HALF_WIDTH_M, Z_AXIS_MAX_M))
    ax.view_init(elev=23, azim=-58)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")


def style_slice_axes(ax) -> None:
    ax.set_xlim(-AXIS_HALF_WIDTH_M, AXIS_HALF_WIDTH_M)
    ax.set_ylim(0.0, Z_AXIS_MAX_M)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#D0D5DA", linewidth=0.55)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")


def main() -> None:
    full = np.load(FULL_PATH)
    scan = np.load(SCAN_PATH)

    points = full["points_xyz"]
    labels = full["fragment_ids"]
    scan_points = scan["points_xyz"]
    scan_labels = scan["fragment_ids"]
    retained_full_indices = np.unique(scan["exterior_source_indices"])
    retained_mask = np.zeros(len(points), dtype=bool)
    retained_mask[retained_full_indices] = True

    rng = np.random.default_rng(15)
    idx_full = rng.choice(len(points), size=min(45_000, len(points)), replace=False)
    idx_scan = rng.choice(len(scan_points), size=min(45_000, len(scan_points)), replace=False)

    full_slice = np.abs(points[:, 0]) <= SLICE_HALF_WIDTH_M
    scan_slice = np.abs(scan_points[:, 0]) <= SLICE_HALF_WIDTH_M
    retained_slice = full_slice & retained_mask
    removed_slice = full_slice & ~retained_mask
    idx_removed = np.where(removed_slice)[0]
    idx_retained = np.where(retained_slice)[0]
    idx_scan_slice = np.where(scan_slice)[0]
    if len(idx_removed) > 12000:
        idx_removed = rng.choice(idx_removed, 12000, replace=False)
    if len(idx_retained) > 12000:
        idx_retained = rng.choice(idx_retained, 12000, replace=False)
    if len(idx_scan_slice) > 12000:
        idx_scan_slice = rng.choice(idx_scan_slice, 12000, replace=False)

    fig = plt.figure(figsize=(12.2, 8.0), dpi=180)
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.scatter(points[idx_full, 0], points[idx_full, 1], points[idx_full, 2], c=labels[idx_full], s=0.35, cmap="tab20", linewidths=0)
    ax1.set_title("Original full labelled point cloud", pad=8)
    style_axes(ax1)

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2.scatter(
        scan_points[idx_scan, 0],
        scan_points[idx_scan, 1],
        scan_points[idx_scan, 2],
        c=scan_labels[idx_scan],
        s=0.45,
        cmap="tab20",
        linewidths=0,
    )
    ax2.set_title("Exterior-only scan after occlusion filtering", pad=8)
    style_axes(ax2)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(
        points[idx_retained, 1],
        points[idx_retained, 2],
        c="#86A8C8",
        s=1.0,
        alpha=0.45,
        linewidths=0,
        label="retained exterior source samples",
    )
    ax3.scatter(
        points[idx_removed, 1],
        points[idx_removed, 2],
        c="#D84A3A",
        s=1.8,
        alpha=0.75,
        linewidths=0,
        label="removed hidden/interior source samples",
    )
    ax3.set_title(f"Half-cut y-z section at x=0 before filtering (|x| <= {SLICE_HALF_WIDTH_M:.2f} m)")
    style_slice_axes(ax3)
    ax3.legend(loc="upper right", fontsize=7, frameon=True)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(
        scan_points[idx_scan_slice, 1],
        scan_points[idx_scan_slice, 2],
        c=scan_labels[idx_scan_slice],
        s=1.2,
        cmap="tab20",
        linewidths=0,
    )
    ax4.set_title("Same half-cut section after exterior-only filtering")
    style_slice_axes(ax4)

    retained_fraction = retained_full_indices.size / len(points)
    fig.text(
        0.50,
        0.022,
        (
            "All panels use matched metre scales. Bottom half-cut sections verify the filtering: red full-cloud source "
            f"samples are not retained by the exterior scan ({retained_fraction:.1%} retained overall)."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.93, bottom=0.075, wspace=0.12, hspace=0.28)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
