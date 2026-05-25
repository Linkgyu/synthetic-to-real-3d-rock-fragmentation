"""Particle size distribution utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def equivalent_spherical_diameter(volume_m3: np.ndarray) -> np.ndarray:
    """Convert fragment volumes to equivalent spherical diameters in metres."""

    volume = np.asarray(volume_m3, dtype=float)
    return (6.0 * volume / np.pi) ** (1.0 / 3.0)


def cumulative_psd(diameter_m: np.ndarray, volume_m3: np.ndarray) -> pd.DataFrame:
    """Compute volume-weighted cumulative passing PSD."""

    diameter = np.asarray(diameter_m, dtype=float)
    volume = np.asarray(volume_m3, dtype=float)
    order = np.argsort(diameter)
    diameter = diameter[order]
    volume = volume[order]
    fraction = volume / np.clip(volume.sum(), 1e-12, None)
    return pd.DataFrame(
        {
            "diameter_m": diameter,
            "diameter_mm": diameter * 1000.0,
            "volume_m3": volume,
            "volume_fraction": fraction,
            "cumulative_passing_pct": np.cumsum(fraction) * 100.0,
        }
    )


def percentile_size(psd_df: pd.DataFrame, passing_pct: float) -> float:
    """Interpolate particle size in millimetres at a passing percentage."""

    return float(
        np.interp(
            passing_pct,
            psd_df["cumulative_passing_pct"].to_numpy(),
            psd_df["diameter_mm"].to_numpy(),
        )
    )

