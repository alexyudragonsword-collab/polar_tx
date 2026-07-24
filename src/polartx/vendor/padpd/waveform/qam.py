# Vendored from PA_DPD@44f9ee99: src/padpd/waveform/qam.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Gray-coded square QAM mapping/demapping.

Supports 4/16/64/256/1024/4096-QAM (square constellations, even number of
bits per symbol). Constellations are normalized to unit average power.
"""

from __future__ import annotations

import numpy as np

_SUPPORTED = (4, 16, 64, 256, 1024, 4096)


def _binary_to_gray(b: np.ndarray) -> np.ndarray:
    return b ^ (b >> 1)


def _pam_levels(m_side: int) -> np.ndarray:
    """PAM amplitude levels (..., -3, -1, 1, 3, ...) indexed by Gray code."""
    binary_index = np.arange(m_side)
    gray_index = _binary_to_gray(binary_index)
    # levels[gray] = amplitude of the point whose Gray label is `gray`
    levels = np.empty(m_side)
    levels[gray_index] = 2 * binary_index - (m_side - 1)
    return levels


def qam_constellation(order: int) -> np.ndarray:
    """Return the Gray-coded constellation, indexed by symbol integer.

    constellation[k] is the complex point for the integer label k, where the
    high bits of k select the I (real) PAM level and the low bits select Q.
    Normalized to unit average power.
    """
    if order not in _SUPPORTED:
        raise ValueError(f"QAM order must be one of {_SUPPORTED}, got {order}")
    m_side = int(np.sqrt(order))
    levels = _pam_levels(m_side)
    i_idx, q_idx = np.divmod(np.arange(order), m_side)
    points = levels[i_idx] + 1j * levels[q_idx]
    scale = np.sqrt((np.abs(points) ** 2).mean())
    return points / scale


def qam_modulate(symbols: np.ndarray, order: int) -> np.ndarray:
    """Map integer symbols in [0, order) to complex constellation points."""
    constellation = qam_constellation(order)
    symbols = np.asarray(symbols)
    if symbols.min() < 0 or symbols.max() >= order:
        raise ValueError("symbol labels out of range")
    return constellation[symbols]


def qam_demodulate(points: np.ndarray, order: int) -> np.ndarray:
    """Hard-decision demap complex points to integer symbol labels.

    Uses per-axis PAM slicing (O(N) instead of an O(N*order) full search),
    which matters for 4096-QAM.
    """
    m_side = int(np.sqrt(order))
    levels = _pam_levels(m_side)
    # gray label -> amplitude is `levels`; build amplitude-sorted lookup
    sort_idx = np.argsort(levels)  # gray labels ordered by amplitude
    constellation_scale = np.sqrt((2 / 3) * (order - 1))  # unnormalized RMS
    pts = np.asarray(points) * constellation_scale

    def slice_axis(vals: np.ndarray) -> np.ndarray:
        # nearest PAM level index on the amplitude axis
        idx = np.clip(np.round((vals + (m_side - 1)) / 2), 0, m_side - 1)
        amp_order = idx.astype(int)
        return sort_idx[amp_order]

    i_lab = slice_axis(pts.real)
    q_lab = slice_axis(pts.imag)
    return i_lab * m_side + q_lab
