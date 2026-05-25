"""P80 estimation and error helpers."""

from __future__ import annotations


def p80_error_mm(estimated_p80_mm: float, ground_truth_p80_mm: float) -> float:
    """Return signed P80 error in millimetres."""

    return float(estimated_p80_mm - ground_truth_p80_mm)


def p80_relative_error_pct(estimated_p80_mm: float, ground_truth_p80_mm: float) -> float:
    """Return signed relative P80 error in percent."""

    if ground_truth_p80_mm == 0:
        raise ValueError("ground_truth_p80_mm must be non-zero.")
    return 100.0 * p80_error_mm(estimated_p80_mm, ground_truth_p80_mm) / ground_truth_p80_mm

