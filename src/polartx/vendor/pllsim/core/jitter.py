# Vendored from pll_simulator@931cfaf: src/pllsim/core/jitter.py
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

import math
from dataclasses import dataclass

import numpy as np

TWOPI = 2.0 * np.pi

#: 10*log10(2).  The gap between the two dBc conventions below, and the
#: single most common silent error in phase-noise numbers: read a figure
#: under the wrong one and every jitter derived from it is off by sqrt(2).
HALF_POWER_DB = 10.0 * math.log10(2.0)

#: Above this RMS phase the small-angle picture stops holding: the carrier is
#: measurably depressed, so integrated phase power and dBc-below-carrier are
#: no longer the same statement.  0.1 rad is 5.73 deg, about -20 dBc.
SMALL_ANGLE_RAD = 0.1

# np.trapezoid is the numpy>=2 name for np.trapz, and this was the one call
# that silently required numpy 2 while pyproject declared >=1.24.  The
# Android build pins numpy 1.x (no 2.x wheels for its Python), so the floor
# has to be real: resolve whichever name this numpy provides.
_trapezoid = vars(np).get("trapezoid") or vars(np)["trapz"]   # 2.x / 1.x


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
    return float(_trapezoid(si, fi))


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


# --------------------------------------------------------------------------
# Unit conversion between the three ways one integrated phase noise is quoted


@dataclass(frozen=True)
class PhaseNoiseUnits:
    """One RMS phase deviation, expressed every way a datasheet quotes it.

    All five numbers describe the same physical quantity; nothing here
    integrates anything.  The pivot is `rad`, and both dBc figures are
    reported because *which one a source means is not deducible from the
    number itself* -- see the two fields below.
    """

    f0: float
    """Carrier [Hz].  Only the jitter depends on it; degrees and dBc do not,
    which is why jitter is the wrong axis for comparing sources at different
    carriers and dBc is the right one."""

    rad: float
    deg: float
    jitter_s: float

    ipn_dbc_dsb: float
    """10*log10(sigma^2): both sidebands, i.e. the whole phase power.  The
    convention most synthesizer datasheets and phase-noise analyzers quote."""

    ipn_dbc_ssb: float
    """10*log10(sigma^2 / 2): one sideband.  What `ipn_dbc()` above returns
    and what `AnalysisResult.ipn_dbc` carries, so this is the field to
    compare against anything else in this package.  Exactly
    HALF_POWER_DB (3.0103 dB) below `ipn_dbc_dsb`."""

    @property
    def jitter_fs(self) -> float:
        return 1e15 * self.jitter_s

    @property
    def jitter_ps(self) -> float:
        return 1e12 * self.jitter_s

    @property
    def small_angle(self) -> bool:
        """False once the dBc figures stop meaning what they appear to."""
        return self.rad <= SMALL_ANGLE_RAD


def convert_phase_noise(f0: float, *,
                        rad: float | None = None,
                        deg: float | None = None,
                        jitter_s: float | None = None,
                        jitter_fs: float | None = None,
                        ipn_dbc_dsb: float | None = None,
                        ipn_dbc_ssb: float | None = None) -> PhaseNoiseUnits:
    """Convert between degrees, jitter and integrated dBc at carrier `f0`.

    Give exactly one quantity; the rest follow from

        sigma [rad] -> deg          x 180/pi
                    -> jitter [s]    / (2*pi*f0)
                    -> dBc (DSB)     20*log10(sigma)
                    -> dBc (SSB)     20*log10(sigma) - 10*log10(2)

    Requiring exactly one is not pedantry.  The natural GUI for this is three
    fields that update each other, and letting two be "given" at once is how
    such a form ends up quietly resolving a contradiction in favour of
    whichever branch happens to be checked first.
    """
    given = {k: v for k, v in (("rad", rad), ("deg", deg),
                               ("jitter_s", jitter_s), ("jitter_fs", jitter_fs),
                               ("ipn_dbc_dsb", ipn_dbc_dsb),
                               ("ipn_dbc_ssb", ipn_dbc_ssb))
             if v is not None}
    if len(given) != 1:
        raise ValueError(
            f"give exactly one quantity, got {sorted(given) or 'none'}")
    if not (math.isfinite(f0) and f0 > 0.0):
        raise ValueError(f"f0 must be a positive frequency, got {f0!r}")

    name, value = next(iter(given.items()))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")

    if name in ("rad", "deg", "jitter_s", "jitter_fs") and value <= 0.0:
        # zero is not merely an edge case: it maps to -inf dBc, and a form
        # that accepts it displays "-inf" where a reader expects a number
        raise ValueError(f"{name} must be positive, got {value!r}")

    if name == "rad":
        sigma = value
    elif name == "deg":
        sigma = math.radians(value)
    elif name == "jitter_s":
        sigma = value * TWOPI * f0
    elif name == "jitter_fs":
        sigma = value * 1e-15 * TWOPI * f0
    elif name == "ipn_dbc_dsb":
        sigma = math.sqrt(10.0 ** (value / 10.0))
    else:                                            # ipn_dbc_ssb
        sigma = math.sqrt(2.0 * 10.0 ** (value / 10.0))

    dsb = 20.0 * math.log10(sigma)
    return PhaseNoiseUnits(
        f0=float(f0),
        rad=sigma,
        deg=math.degrees(sigma),
        jitter_s=sigma / (TWOPI * f0),
        ipn_dbc_dsb=dsb,
        ipn_dbc_ssb=dsb - HALF_POWER_DB,
    )
