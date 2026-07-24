# Vendored from PA_DPD@44f9ee99: src/padpd/cfr.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Crest Factor Reduction by iterative clipping and filtering (ICF).

DPD can only linearize up to the PA's maximum output: OFDM peaks (PAPR
~10-12 dB) that drive the PA past its AM-AM turnover are unrecoverable
and make ILA diverge. Industrial transmit chains therefore run CFR
before DPD, trading a bounded amount of in-band EVM for several dB of
PAPR so that the whole envelope stays inside the PA's invertible region.

Iterative clipping and filtering: hard-clip the envelope at the target
level, then brick-wall filter to the signal bandwidth to remove the
clipping splatter. Filtering partially regrows the peaks, so the pair of
steps is iterated.
"""

from __future__ import annotations

import numpy as np


def cfr_clip_filter(x: np.ndarray, target_papr_db: float, fs: float,
                    bandwidth_hz: float, n_iterations: int = 4) -> np.ndarray:
    """Reduce PAPR of a baseband signal to ~``target_papr_db``.

    The clip threshold is set from the ORIGINAL signal's RMS so the
    average power (and the downstream PA operating point) is essentially
    preserved. The out-of-band clipping products are removed with an FFT
    brick-wall filter over +/- bandwidth_hz/2 each iteration.
    """
    x = np.asarray(x, dtype=complex)
    rms = np.sqrt(np.mean(np.abs(x) ** 2))
    threshold = rms * 10 ** (target_papr_db / 20)

    freq = np.fft.fftfreq(len(x), d=1 / fs)
    in_band = np.abs(freq) <= bandwidth_hz / 2

    y = x
    for _ in range(n_iterations):
        env = np.abs(y)
        over = env > threshold
        if not over.any():
            break
        y = np.where(over, threshold * y / np.where(over, env, 1), y)
        spectrum = np.fft.fft(y)
        spectrum[~in_band] = 0
        y = np.fft.ifft(spectrum)
    return y
