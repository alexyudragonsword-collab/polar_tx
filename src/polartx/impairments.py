"""Path impairments injected by the chain."""
from __future__ import annotations

import numpy as np

from .vendor.padpd.data.align import _fractional_advance


def fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    """Delay x by a (possibly fractional) number of samples.

    FFT phase-ramp implementation (circular; edge samples affected for
    the fractional part only), exact for band-limited signals.  Works on
    real or complex input and preserves realness.
    """
    x = np.asarray(x)
    if delay_samples == 0.0:
        return x.copy()
    y = _fractional_advance(x.astype(complex), -delay_samples)
    return y.real if np.isrealobj(x) else y


def zoh_hold(x: np.ndarray, hold: int, offset: int = 0) -> np.ndarray:
    """Sample-and-hold every `hold`-th sample (DPA/phase update clock
    slower than the baseband grid) — produces the ZOH images at
    multiples of fs/hold.  offset staggers the update instants
    (interleaved DPA banks)."""
    if hold <= 1:
        return np.asarray(x).copy()
    x = np.asarray(x)
    idx = ((np.arange(x.size) - offset) // hold) * hold + offset
    return x[np.clip(idx, 0, x.size - 1)]
