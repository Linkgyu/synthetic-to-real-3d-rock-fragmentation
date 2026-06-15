from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEG_PATH = ROOT / "data" / "labels" / "synthetic_rockpile_exterior_surface_segmentation.npz"
METRICS_PATH = ROOT / "outputs" / "tables" / "surface_segmentation_metrics.csv"
OUT_PATH = ROOT / "outputs" / "figures" / "synthetic_rockpile_surface_segmentation_split.png"


def load_metrics() -> dict[str, float]:
    with METRICS_PATH.open("r", newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    return {key: float(value) for key, value in row.items()}


def main() -> None:
    data = np.load(SEG_PATH)
    points = data["points_xyz"].astype(np.float32)
    labels = data["labels_surface"]
    metrics = load_metrics()

    rng = np.random.default_rng(42)
    idx = rng.choice(len(points), size=min(45_000, len(points)), replace=False)
    sample = points[idx]
    plot_labels = labels[idx]

    fig = plt.figure(figsize=(8.6, 7.1), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c=plot_labels, s=0.35, cmap="tab20", linewidths=0)
    ax.set_title(
        "Surface segmentation after marker split "
        f"({int(metrics['n_predicted_clusters'])} clusters, ARI={metrics['adjusted_rand_index']:.3f})"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_zlim(0.0, 1.0)
    ax.set_box_aspect((2.5, 2.5, 1.55))
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
