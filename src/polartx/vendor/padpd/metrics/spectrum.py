# Vendored from PA_DPD@44f9ee99: src/padpd/metrics/spectrum.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Power spectral density and simplified 802.11-style spectral mask check."""

from __future__ import annotations

import numpy as np
from scipy import signal as sig


def psd(x: np.ndarray, fs: float, nfft: int = 4096):
    """Welch PSD, fftshifted. Returns (freqs_hz, psd_db) with 0 dB = peak."""
    freqs, pxx = sig.welch(x, fs=fs, nperseg=min(nfft, len(x)),
                           return_onesided=False, detrend=False)
    order = np.argsort(freqs)
    freqs, pxx = freqs[order], pxx[order]
    pxx_db = 10 * np.log10(pxx / pxx.max())
    return freqs, pxx_db


def default_wifi_mask(bandwidth_hz: float):
    """Simplified 802.11-style transmit mask, scaled to channel bandwidth.

    Piecewise-linear (frequency offset, dBr) breakpoints, one side; the
    mask is symmetric. Follows the 802.11ax/be 0/-20/-28/-40 dBr template
    (e.g. 80 MHz: 39.5 / 40.5 / 60 / 80 MHz), scaled by bandwidth.
    Engineering simplification, not the exact standard clause.
    """
    bw = bandwidth_hz
    return np.array([
        (0.000 * bw, 0.0),
        (0.494 * bw, 0.0),
        (0.506 * bw, -20.0),
        (0.750 * bw, -28.0),
        (1.000 * bw, -40.0),
        (5.000 * bw, -40.0),
    ])


def check_mask(freqs: np.ndarray, psd_db: np.ndarray, mask: np.ndarray):
    """Compare a PSD (dB rel. peak) against a symmetric mask.

    Returns (passes, worst_margin_db, mask_db_at_freqs). Positive margin =
    the PSD is below the mask everywhere.
    """
    mask_db = np.interp(np.abs(freqs), mask[:, 0], mask[:, 1],
                        right=mask[-1, 1])
    margin = mask_db - psd_db
    worst = float(margin.min())
    return worst >= 0, worst, mask_db
