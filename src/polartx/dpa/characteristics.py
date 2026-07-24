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


def ampm_curve(ampm_deg_poly: tuple, r: np.ndarray) -> np.ndarray:
    """AM-PM [rad] vs normalized amplitude; poly coeffs low-order-first
    in degrees, referenced so ampm(0) contributes its constant term."""
    r = np.asarray(r, dtype=float)
    if not ampm_deg_poly:
        return np.zeros_like(r)
    deg = np.polyval(tuple(ampm_deg_poly)[::-1], r)
    return np.deg2rad(deg)
