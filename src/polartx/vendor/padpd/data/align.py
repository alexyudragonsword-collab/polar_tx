# Vendored from PA_DPD@44f9ee99: src/padpd/data/align.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Time alignment of PA input/output captures.

Measured or EDA-exported data often has an unknown bulk delay (and complex
gain) between the stimulus and the capture. Behavioral-model fitting
assumes sample-aligned x/y, so run :func:`align_delay` first. (OpenDPD
datasets ship pre-aligned; Cadence envelope exports and lab captures
usually do not.)
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sig


def _fractional_advance(y: np.ndarray, frac: float) -> np.ndarray:
    """Advance y by a fractional number of samples via an FFT phase ramp.

    Circular by construction; only the few edge samples are affected for
    |frac| < 1.
    """
    freq = np.fft.fftfreq(len(y))
    return np.fft.ifft(np.fft.fft(y) * np.exp(2j * np.pi * freq * frac))


def align_delay(x: np.ndarray, y: np.ndarray, max_lag: int = 4096):
    """Estimate and remove the bulk delay of ``y`` relative to ``x``.

    The integer delay is the argmax of the complex cross-correlation
    within ``+/- max_lag`` samples (positive = y lags x). A residual
    fractional delay is then estimated by parabolic interpolation of the
    correlation peak and, if larger than 0.02 samples, removed with an
    FFT phase ramp.

    Returns ``(x_aligned, y_aligned, info)`` where the aligned arrays are
    the overlapping region and ``info`` holds ``{"lag"``: integer part,
    ``"lag_total"``: float total delay, ``"gain"``: least-squares complex
    gain with y ≈ gain * x after alignment``}``.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    corr = sig.correlate(y, x, mode="full", method="fft")
    lags = sig.correlation_lags(len(y), len(x), mode="full")
    window = np.where(np.abs(lags) <= max_lag)[0]
    peak = window[np.argmax(np.abs(corr[window]))]
    lag = int(lags[peak])

    # Parabolic interpolation of |corr| around the peak -> fractional lag.
    frac = 0.0
    if 0 < peak < len(corr) - 1:
        c_m, c_0, c_p = np.abs(corr[peak - 1: peak + 2])
        denom = c_m - 2 * c_0 + c_p
        if denom != 0:
            frac = float(np.clip(0.5 * (c_m - c_p) / denom, -0.5, 0.5))

    if lag >= 0:
        y_a = y[lag:]
        x_a = x[: len(y_a)]
    else:
        x_a = x[-lag:]
        y_a = y[: len(x_a)]
    n = min(len(x_a), len(y_a))
    x_a, y_a = x_a[:n], y_a[:n]

    if abs(frac) > 0.02:
        y_a = _fractional_advance(y_a, frac)

    # Refine with the cross-spectrum phase slope: parabolic peak
    # interpolation is biased for wide-band or nonlinearly distorted
    # signals (0.1-0.3 sample residual on PA outputs), and a residual
    # fractional delay turns into a phase ramp across subcarriers that
    # caps constellation EVM around -20 dB. The phase-vs-frequency slope
    # of X* Y over the occupied band is a far more accurate estimator.
    resid = _phase_slope_delay(x_a, y_a)
    if abs(resid) > 0.002:
        y_a = _fractional_advance(y_a, resid)
        frac += resid

    gain = complex(np.vdot(x_a, y_a) / np.vdot(x_a, x_a))
    return x_a, y_a, {"lag": lag, "lag_total": lag + frac, "gain": gain}


def _phase_slope_delay(x: np.ndarray, y: np.ndarray) -> float:
    """Residual delay of y vs x from the cross-spectrum phase slope.

    Weighted LS fit of unwrapped angle(X* Y) against frequency over the
    occupied band (bins with meaningful power); returns the delay in
    samples (positive = y still lags x).
    """
    n = len(x)
    xf = np.fft.fft(x)
    yf = np.fft.fft(y)
    cross = np.conj(xf) * yf
    w = np.abs(cross)
    band = w > 0.05 * w.max()
    if band.sum() < 8:
        return 0.0
    f = np.fft.fftfreq(n)
    order = np.argsort(f[band])
    fb = f[band][order]
    ph = np.unwrap(np.angle(cross[band][order]))
    wb = w[band][order]
    fc = fb - np.average(fb, weights=wb)
    denom = np.sum(wb * fc ** 2)
    if denom == 0:
        return 0.0
    slope = np.sum(wb * fc * (ph - np.average(ph, weights=wb))) / denom
    return float(np.clip(-slope / (2 * np.pi), -1.0, 1.0))
