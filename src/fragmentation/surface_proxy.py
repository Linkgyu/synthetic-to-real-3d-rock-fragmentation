"""Surface-scan size proxies for PSD estimation.

Exterior-only scans do not contain closed fragment volumes. These helpers use
robust geometric spans as a transparent proxy instead of convex hull volume,
which is unstable when clusters are partial or accidentally merged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _robust_pca_spans(points: np.ndarray, lower_pct: float = 2.0, upper_pct: float = 98.0) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    if len(points) < 3:
        return np.zeros(3, dtype=float)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh.T
    lo = np.percentile(projected, lower_pct, axis=0)
    hi = np.percentile(projected, upper_pct, axis=0)
    return np.maximum(hi - lo, 0.0)


def _sampled_pairwise_distance(points: np.ndarray, percentile: float = 90.0, max_points: int = 512, seed: int = 0) -> float:
    if len(points) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    if len(points) > max_points:
        idx = rng.choice(len(points), size=max_points, replace=False)
        sample = points[idx]
    else:
        sample = points
    diff = sample[:, None, :] - sample[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    tri = dist[np.triu_indices(len(sample), k=1)]
    return float(np.percentile(tri, percentile)) if len(tri) else 0.0


def estimate_surface_size_proxy(
    points_xyz: np.ndarray,
    cluster_labels: np.ndarray,
    min_points: int = 40,
    pairwise_percentile: float = 90.0,
) -> pd.DataFrame:
    """Estimate fragment size from exterior cluster surface geometry.

    The reported volume is a proxy volume computed from the proxy diameter as
    pi/6 * d^3. It is appropriate for sensitivity comparisons, but it should not
    be interpreted as a physically measured closed-mesh volume.
    """

    points = np.asarray(points_xyz, dtype=float)
    labels = np.asarray(cluster_labels, dtype=int)
    rows = []
    for label in sorted(set(labels)):
        if label < 0:
            continue
        idx = np.flatnonzero(labels == label)
        if len(idx) < min_points:
            continue
        cluster = points[idx]
        spans = _robust_pca_spans(cluster)
        pairwise_d = _sampled_pairwise_distance(cluster, percentile=pairwise_percentile, seed=int(label) + 17)
        pca_d = float(spans[0])
        proxy_d = float(np.median([pairwise_d, pca_d, max(spans[0], 1.15 * spans[1])]))
        proxy_volume = np.pi / 6.0 * proxy_d**3
        rows.append(
            {
                "cluster_id": int(label),
                "n_points": int(len(idx)),
                "diameter_proxy_m": proxy_d,
                "diameter_proxy_mm": proxy_d * 1000.0,
                "pca_span_1_m": float(spans[0]),
                "pca_span_2_m": float(spans[1]),
                "pca_span_3_m": float(spans[2]),
                "pairwise_d90_m": pairwise_d,
                "proxy_volume_m3": proxy_volume,
            }
        )
    return pd.DataFrame(rows)
