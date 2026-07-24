"""DPA AM-AM / AM-PM characteristics versus normalized amplitude.

amam options (DPAConfig.amam):
- "ideal": amplitude linear in code (ideal SCPA law).
- ("rapp", p, drive): Rapp soft compression r_out = d*r/(1+(d*r)^(2p))^(1/2p),
  normalized so full scale maps to full scale; p ~ 1-3 CMOS-PA-like.
- ("lut", r_in, r_out): measured/imported AM-AM (e.g. via the vendored
  padpd hb_import loaders), interpolated.

AM-PM is a polynomial in normalized amplitude giving degrees
(ampm_deg_poly, low-order-first), the usual capacitance-modulation shape.
"""
from __future__ import annotations

import numpy as np


def amam_curve(amam, r: np.ndarray) -> np.ndarray:
    """Map normalized input amplitude r in [0,1] -> normalized output."""
    r = np.asarray(r, dtype=float)
    if amam == "ideal":
        return r
    kind = amam[0]
    if kind == "rapp":
        _, p, drive = amam
        x = drive * r
        y = x / (1.0 + x ** (2.0 * p)) ** (1.0 / (2.0 * p))
        y_fs = drive / (1.0 + drive ** (2.0 * p)) ** (1.0 / (2.0 * p))
        return y / y_fs
    if kind == "lut":
        _, r_in, r_out = amam
        y = np.interp(r, np.asarray(r_in, float), np.asarray(r_out, float))
        return y / np.interp(1.0, np.asarray(r_in, float),
                             np.asarray(r_out, float))
    raise ValueError(f"unknown amam spec: {amam!r}")


def efficiency_curve(eff_spec, x: np.ndarray) -> np.ndarray:
    """Drain efficiency vs normalized output amplitude x = n/N.

    ("scpa", gamma, eta_peak): ideal class-D SCPA law (Yoo & Walling,
    JSSC 2011) — output power ∝ x², switched-capacitor charging loss
    ∝ x(1−x), so η(x) = η_peak · x² / (x² + γ·x(1−x)).  γ sets the
    backoff rolloff (γ = 0.67 → 60% of peak at half amplitude); η_peak
    absorbs switch/matching losses.
    ("classb", eta_peak): η = η_peak · x (linear-PA comparison line).
    ("doherty", gamma, eta_peak, bo_db): SCPA base law with a second
    efficiency peak at bo_db backoff (load modulation) — the digital-
    Doherty enhancement (Borokhovich, RFIC 2026): disabling half the
    cells at backoff modulates the combiner load, holding efficiency up
    near the backoff point.  Phenomenological (matches the measured
    double-hump), not an ideal-class-B derivation.
    ("lut", x_pts, eta_pts): measured curve, interpolated.
    """
    x = np.asarray(x, dtype=float)
    kind = eff_spec[0]
    if kind == "scpa":
        _, gamma, eta_peak = eff_spec
        num = x * x
        den = num + gamma * x * (1.0 - x)
        return eta_peak * np.divide(num, den, out=np.zeros_like(x),
                                    where=den > 0)
    if kind == "classb":
        return eff_spec[1] * x
    if kind == "doherty":
        _, gamma, eta_peak, bo_db = eff_spec
        x_bo = 10.0 ** (-bo_db / 20.0)              # backoff peak location
        # below the backoff point: SCPA law rescaled to peak at x_bo
        low = efficiency_curve(("scpa", gamma, eta_peak),
                               np.clip(x / x_bo, 0.0, 1.0))
        # above: load modulation holds efficiency up, classic Doherty dip
        # (parabola: eta_peak at x_bo and at x=1, ~15% dip mid-way)
        t = np.clip((x - x_bo) / (1.0 - x_bo), 0.0, 1.0)
        high = eta_peak * (1.0 - 0.15 * 4.0 * t * (1.0 - t))
        return np.where(x <= x_bo, low, high)
    if kind == "lut":
        _, xp, ep = eff_spec
        return np.interp(x, np.asarray(xp, float), np.asarray(ep, float))
    raise ValueError(f"unknown efficiency spec: {eff_spec!r}")


def ampm_curve(ampm_deg_poly: tuple, r: np.ndarray) -> np.ndarray:
    """AM-PM [rad] vs normalized amplitude; poly coeffs low-order-first
    in degrees, referenced so ampm(0) contributes its constant term."""
    r = np.asarray(r, dtype=float)
    if not ampm_deg_poly:
        return np.zeros_like(r)
    deg = np.polyval(tuple(ampm_deg_poly)[::-1], r)
    return np.deg2rad(deg)
