"""Open-loop DTC gain/INL calibration from demodulated phase error.

A feedback receiver (or factory ATE) observes the transmitted phase and
compares it with the command; binning the error against the DTC code
yields the gain error (LS slope) and the INL shape (per-segment means, a
K-segment LUT — the same structure as the pllsim in-loop LUTCal, fitted
offline here).  apply_dtc_correction() installs both on the modulator,
which then pre-corrects the command before quantization.
"""
from __future__ import annotations

import numpy as np

TWOPI = 2.0 * np.pi


def fit_dtc_correction(pm, phase_cmd: np.ndarray, fs_bb: float, *,
                       n_seg: int = 32, noise: bool = False,
                       seed: int = 0) -> dict:
    """Training pass: run the modulator on phase_cmd, fit gain + INL.

    Returns {"gain_hat", "inl_lut_rad", "inl_lut_x", "err_rms_before"}.
    Use a stimulus that sweeps the full code range (a CW frequency
    offset does; so does wideband OFDM polar phase).
    """
    res = pm.modulate(phase_cmd, fs_bb, noise=noise, seed=seed)
    m = pm.cfg.range_rad
    err = np.mod(res.phase_out - phase_cmd + 0.5 * m, m) - 0.5 * m
    x = np.mod(phase_cmd, m) / m                     # normalized command

    # gain: LS slope of err vs (x - 1/2)*m (bipolar around mid-range)
    xc = (x - 0.5) * m
    gain_hat = float(np.dot(err - err.mean(), xc - xc.mean())
                     / np.dot(xc - xc.mean(), xc - xc.mean()))
    resid = err - gain_hat * xc

    edges = np.linspace(0.0, 1.0, n_seg + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_seg - 1)
    lut = np.zeros(n_seg)
    for b in range(n_seg):
        sel = idx == b
        if sel.sum() >= 4:
            lut[b] = resid[sel].mean()
    lut -= lut.mean()                                # static offset is free
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {"gain_hat": gain_hat, "inl_lut_rad": lut,
            "inl_lut_x": centers,
            "err_rms_before": float(np.std(err))}


def apply_dtc_correction(pm, fit: dict) -> None:
    """Install the fitted correction on a DTCPhaseModulator.

    The gain LUT correction composes with any previous one (iterate
    fit/apply twice for the last dB)."""
    pm.gain_hat += fit["gain_hat"] * (1.0 + pm.gain_hat)
    if pm.inl_lut_rad is None:
        pm.inl_lut_rad = fit["inl_lut_rad"].copy()
        pm.inl_lut_x = fit["inl_lut_x"].copy()
    else:
        pm.inl_lut_rad = pm.inl_lut_rad + np.interp(
            pm.inl_lut_x, fit["inl_lut_x"], fit["inl_lut_rad"])
