# Vendored from PA_DPD@44f9ee99: src/padpd/metrics/ccdf.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""CCDF of the instantaneous power (peak-to-average statistics).

The complementary cumulative distribution function of instantaneous
power relative to average power is the standard way to characterize a
waveform's peak behavior (beyond the single-number PAPR) and to show the
effect of CFR.
"""

from __future__ import annotations

import numpy as np


def ccdf(x: np.ndarray, max_db: float | None = None, n_points: int = 200):
    """CCDF of instantaneous power above average.

    Returns ``(level_db, prob)`` where ``prob[i]`` is the probability
    that the instantaneous power exceeds the average power by more than
    ``level_db[i]`` dB.
    """
    p = np.abs(np.asarray(x)) ** 2
    rel_db = 10 * np.log10(np.maximum(p / p.mean(), 1e-300))
    if max_db is None:
        max_db = rel_db.max()
    level_db = np.linspace(0.0, max_db, n_points)
    sorted_db = np.sort(rel_db)
    # P(rel > level) via binary search on the sorted samples
    idx = np.searchsorted(sorted_db, level_db, side="right")
    prob = 1.0 - idx / len(sorted_db)
    return level_db, prob
