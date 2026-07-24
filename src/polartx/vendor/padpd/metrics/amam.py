# Vendored from PA_DPD@44f9ee99: src/padpd/metrics/amam.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""AM-AM / AM-PM characteristic extraction from input/output IQ data."""

from __future__ import annotations

import numpy as np


def am_am_am_pm(x: np.ndarray, y: np.ndarray, n_bins: int = 60) -> dict:
    """Extract AM-AM and AM-PM characteristics.

    Returns raw per-sample scatter data and amplitude-binned averages:
    {"r_in", "gain", "phase_deg", "bin_r", "bin_gain", "bin_phase_deg"}.
    """
    r = np.abs(x)
    valid = r > 1e-8
    r = r[valid]
    ratio = y[valid] / x[valid]
    gain = np.abs(ratio)
    phase = np.angle(ratio, deg=True)

    edges = np.linspace(0, r.max(), n_bins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, n_bins - 1)
    bin_r = np.full(n_bins, np.nan)
    bin_gain = np.full(n_bins, np.nan)
    bin_phase = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            bin_r[b] = r[m].mean()
            bin_gain[b] = gain[m].mean()
            bin_phase[b] = phase[m].mean()
    return {
        "r_in": r, "gain": gain, "phase_deg": phase,
        "bin_r": bin_r, "bin_gain": bin_gain, "bin_phase_deg": bin_phase,
    }
