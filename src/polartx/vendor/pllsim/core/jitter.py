# Vendored from pll_simulator@d7be4712: src/pllsim/core/jitter.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Integrated phase noise and RMS jitter.

Package-wide PSD convention
---------------------------
Internal phase PSDs are **double-sideband** S_phi(f) in rad^2/Hz.
Plots and spot numbers use L(f) = S_phi(f)/2, i.e. dBc/Hz:

    L_dbc(f) = 10*log10(S_phi(f)/2)
    S_phi(f) = 2 * 10^(L_dbc/10)

RMS jitter over [f1, f2]:

    sigma_phi^2 = ∫ S_phi df        [rad^2]
    sigma_t     = sigma_phi / (2*pi*f0)
"""
from __future__ import annotations

import numpy as np

TWOPI = 2.0 * np.pi


def sphi_from_ldbc(l_dbc) -> np.ndarray:
    """dBc/Hz -> double-sideband S_phi [rad^2/Hz]."""
    return 2.0 * 10.0 ** (np.asarray(l_dbc, dtype=float) / 10.0)


def ldbc_from_sphi(s_phi) -> np.ndarray:
    """Double-sideband S_phi [rad^2/Hz] -> L(f) in dBc/Hz."""
    return 10.0 * np.log10(np.maximum(np.asarray(s_phi, dtype=float), 1e-300) / 2.0)


def _band_mask(f: np.ndarray, f1: float, f2: float):
    return (f >= f1) & (f <= f2)


def integrate_pn(f: np.ndarray, s_phi: np.ndarray, f1: float = 1e3, f2: float = 100e6) -> float:
    """∫ S_phi df over [f1, f2] -> phase power [rad^2] (trapezoid on the grid)."""
    f = np.asarray(f, dtype=float)
    s = np.asarray(s_phi, dtype=float)
    m = _band_mask(f, f1, f2)
    fi, si = f[m], s[m]
    # include exact band edges by interpolation in log-f
    if f1 > f[0] and (fi.size == 0 or fi[0] > f1):
        s1 = np.interp(np.log10(f1), np.log10(f), s)
        fi, si = np.insert(fi, 0, f1), np.insert(si, 0, s1)
    if f2 < f[-1] and (fi.size == 0 or fi[-1] < f2):
        s2 = np.interp(np.log10(f2), np.log10(f), s)
        fi, si = np.append(fi, f2), np.append(si, s2)
    if fi.size < 2:
        return 0.0
    return float(np.trapezoid(si, fi))


def ipn_dbc(f: np.ndarray, s_phi: np.ndarray, f1: float = 1e3, f2: float = 100e6) -> float:
    """Integrated phase noise in dBc (single-sideband convention: power/2)."""
    p = integrate_pn(f, s_phi, f1, f2)
    return 10.0 * np.log10(max(p / 2.0, 1e-300))


def rms_jitter_s(f: np.ndarray, s_phi: np.ndarray, f0: float,
                 f1: float = 1e3, f2: float = 100e6) -> float:
    """RMS jitter [s] from double-sideband S_phi at carrier f0."""
    p = integrate_pn(f, s_phi, f1, f2)
    return float(np.sqrt(p) / (TWOPI * f0))


def rms_jitter_fs(f: np.ndarray, s_phi: np.ndarray, f0: float,
                  f1: float = 1e3, f2: float = 100e6) -> float:
    """RMS jitter [fs]."""
    return 1e15 * rms_jitter_s(f, s_phi, f0, f1, f2)
