# Vendored from pll_simulator@d7be4712: src/pllsim/core/spectrum.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Phase-sequence spectrum estimation and spur extraction."""
from __future__ import annotations

import numpy as np
from scipy import signal as sps


def phase_psd(phase_rad: np.ndarray, fs: float, nfft: int | None = None,
              window: str = "hann", detrend: str = "linear"):
    """One-sided PSD of a phase sequence sampled at fs.

    Returns (f, s_phi) with s_phi in rad^2/Hz, directly comparable to the
    linear-model DSB S_phi.  `detrend='linear'` removes residual frequency
    offset (phase ramp) so it does not leak across the spectrum.
    """
    phase_rad = np.asarray(phase_rad, dtype=float)
    n = phase_rad.size
    if nfft is None:
        nfft = min(n, max(1024, n // 8))
    f, pxx = sps.welch(phase_rad, fs=fs, window=window, nperseg=nfft,
                       detrend=detrend, scaling="density")
    return f[1:], pxx[1:]


def periodogram_psd(phase_rad: np.ndarray, fs: float, detrend: str = "linear"):
    """Single full-length periodogram — best frequency resolution for spurs."""
    f, pxx = sps.periodogram(np.asarray(phase_rad, dtype=float), fs=fs,
                             window="blackmanharris", detrend=detrend,
                             scaling="density")
    return f[1:], pxx[1:]


def find_spurs(f: np.ndarray, s_phi: np.ndarray, expected_offsets,
               rbw_bins: int = 5, guard_bins: int = 20) -> dict[float, float]:
    """Spur levels [dBc] at expected offset frequencies.

    Integrates the PSD over a +/-rbw_bins cluster around each offset, subtracts
    the local noise floor (median PSD in the surrounding guard band times the
    cluster width), and converts the residual discrete power to dBc:
        spur_dbc = 10*log10(P_spur/2)   (per sideband, DSB phase convention)
    """
    f = np.asarray(f, dtype=float)
    s = np.asarray(s_phi, dtype=float)
    df = f[1] - f[0] if f.size > 1 else 1.0
    out: dict[float, float] = {}
    for fo in np.atleast_1d(expected_offsets):
        idx = int(round((fo - f[0]) / df))
        if idx < 0 or idx >= f.size:
            out[float(fo)] = float("nan")
            continue
        lo = max(0, idx - rbw_bins)
        hi = min(f.size, idx + rbw_bins + 1)
        p_cluster = np.sum(s[lo:hi]) * df
        glo = max(0, idx - guard_bins)
        ghi = min(f.size, idx + guard_bins + 1)
        noise_bins = np.concatenate([s[glo:lo], s[hi:ghi]])
        noise = float(np.median(noise_bins)) if noise_bins.size else 0.0
        p_spur = p_cluster - noise * (hi - lo) * df
        if p_spur <= 0:
            out[float(fo)] = float("nan")   # below local noise floor
        else:
            out[float(fo)] = float(10.0 * np.log10(p_spur / 2.0))
    return out
