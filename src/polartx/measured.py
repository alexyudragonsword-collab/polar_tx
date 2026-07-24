"""Measured-data DPA modeling via the OpenDPD data path.

Loads real DPA input/output captures (vendored padpd loaders: OpenDPD
dataset folders, Cadence/MATLAB CSV), aligns them, and extracts the
static polar characteristics — binned AM-AM and AM-PM versus input
amplitude — into a polartx DPA model (LUT AM-AM + LUT AM-PM).  The
static-polar NMSE against the capture quantifies how much of the device
is code-static (what the polar DPD LUTs can fix) versus memory (what
needs the Cartesian ILA, cal.memory_dpd).
"""
from __future__ import annotations

import os

import numpy as np

from .dpa import DPA, DPAConfig
from .vendor.padpd.data import align_delay, load_opendpd_dataset
from .vendor.padpd.pa.base import nmse_db

OPENDPD_ENV = "POLARTX_OPENDPD"


def find_opendpd_root() -> str | None:
    """Locate an OpenDPD clone: $POLARTX_OPENDPD, then ../OpenDPD."""
    for cand in (os.environ.get(OPENDPD_ENV),
                 os.path.join(os.path.dirname(__file__),
                              "..", "..", "..", "OpenDPD"),
                 "../OpenDPD"):
        if cand and os.path.isdir(os.path.join(cand, "datasets")):
            return os.path.abspath(cand)
    return None


def extract_polar_characteristics(x: np.ndarray, y: np.ndarray, *,
                                  n_bins: int = 64) -> dict:
    """Aligned, binned static polar model of a measured PA/DPA.

    Returns normalized AM-AM (r_in [0,1] -> r_out [0,1]), AM-PM [deg]
    versus r_in, the linear gain, and the static-polar NMSE: how well
    y is explained by g * lut(|x|) * exp(j(angle(x) + ampm(|x|))).
    """
    x_a, y_a, info = align_delay(np.asarray(x, complex),
                                 np.asarray(y, complex))
    r = np.abs(x_a)
    r_max = r.max()
    rn = r / r_max
    ratio = y_a / np.where(np.abs(x_a) > 1e-12 * r_max, x_a, np.nan)
    gain = np.abs(ratio)
    phase = np.angle(ratio)
    phase = phase - np.nanmedian(phase)          # static rotation is free

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(rn, edges) - 1, 0, n_bins - 1)
    g_bin = np.full(n_bins, np.nan)
    p_bin = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = (idx == b) & np.isfinite(gain)
        if m.sum() >= 8:
            g_bin[b] = np.nanmean(gain[m])
            p_bin[b] = np.nanmean(phase[m])
    centers = 0.5 * (edges[:-1] + edges[1:])
    ok = np.isfinite(g_bin)
    centers, g_bin, p_bin = centers[ok], g_bin[ok], p_bin[ok]

    r_out = centers * g_bin
    r_out = r_out / r_out[-1]
    ampm_deg = np.rad2deg(p_bin)

    # static-polar reconstruction NMSE on the capture
    g_of_r = np.interp(rn, centers, g_bin)
    p_of_r = np.interp(rn, centers, p_bin)
    y_hat = x_a * g_of_r * np.exp(1j * p_of_r)
    return {"r_in": centers, "r_out": r_out, "ampm_deg": ampm_deg,
            "gain": float(np.nanmedian(gain)),
            "static_nmse_db": nmse_db(y_a, y_hat),
            "align": info, "x_aligned": x_a, "y_aligned": y_a}


def dpa_from_measured(x: np.ndarray, y: np.ndarray, *, n_bits: int = 10,
                      n_bins: int = 64, **dpa_kw) -> tuple[DPA, dict]:
    """Build a polartx DPA whose code tables follow the measured device."""
    ch = extract_polar_characteristics(x, y, n_bins=n_bins)
    cfg = DPAConfig(n_bits=n_bits,
                    amam=("lut", ch["r_in"], ch["r_out"]),
                    ampm_lut=(ch["r_in"], ch["ampm_deg"]), **dpa_kw)
    return DPA(cfg), ch


def load_measured_dpa(dataset: str = "DPA_160MHz", *, split: str = "train",
                      n_bits: int = 10, root: str | None = None
                      ) -> tuple[DPA, dict]:
    """OpenDPD dataset folder -> polartx DPA model + extraction report."""
    root = root or find_opendpd_root()
    if root is None:
        raise FileNotFoundError(
            f"OpenDPD clone not found; set ${OPENDPD_ENV} or clone next "
            "to the repo (git clone --depth 1 "
            "https://github.com/lab-emi/OpenDPD.git)")
    d = load_opendpd_dataset(os.path.join(root, "datasets", dataset))
    ds = d[split]
    dpa, ch = dpa_from_measured(ds.x, ds.y, n_bits=n_bits)
    ch["spec"] = d["spec"]
    ch["fs"] = ds.sample_rate_hz
    return dpa, ch
