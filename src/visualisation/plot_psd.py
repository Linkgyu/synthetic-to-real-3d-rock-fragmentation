"""PSD plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_psd(
    psd_df: pd.DataFrame,
    ax: plt.Axes | None = None,
    label: str = "PSD",
    marker: str = "o",
) -> plt.Axes:
    """Plot a cumulative PSD curve."""

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        psd_df["diameter_mm"],
        psd_df["cumulative_passing_pct"],
        marker=marker,
        linewidth=1.6,
        markersize=3,
        label=label,
    )
    ax.set_xlabel("Equivalent spherical diameter [mm]")
    ax.set_ylabel("Cumulative passing [% by volume]")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.25)
    ax.legend()
    return ax

