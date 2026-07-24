# Vendored from PA_DPD@44f9ee99: src/padpd/metrics/aclr.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Adjacent Channel Leakage Ratio."""

from __future__ import annotations

import numpy as np
from scipy import signal as sig


def _band_power(freqs, pxx, f_lo, f_hi) -> float:
    m = (freqs >= f_lo) & (freqs < f_hi)
    return float(pxx[m].sum())


def aclr(x: np.ndarray, fs: float, bandwidth_hz: float,
         nfft: int = 4096) -> dict:
    """ACLR in dBc for the lower and upper adjacent channels.

    Adjacent channels are ``bandwidth_hz`` wide, centered at +/- one
    channel spacing. Requires fs >= 3 * bandwidth (oversampled input).
    """
    if fs < 3 * bandwidth_hz:
        raise ValueError("sample rate too low to observe adjacent channels")
    freqs, pxx = sig.welch(x, fs=fs, nperseg=min(nfft, len(x)),
                           return_onesided=False, detrend=False)
    half = bandwidth_hz / 2
    p_main = _band_power(freqs, pxx, -half, half)
    p_low = _band_power(freqs, pxx, -3 * half, -half)
    p_high = _band_power(freqs, pxx, half, 3 * half)
    return {
        "lower_dbc": 10 * np.log10(p_low / p_main),
        "upper_dbc": 10 * np.log10(p_high / p_main),
    }
