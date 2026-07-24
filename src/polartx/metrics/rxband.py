"""RX-band noise at the duplex offset — the classic FDD polar-TX budget.

At 45-120 MHz duplex spacing the TX's own noise lands in its receiver's
band; -150 dBc/Hz-class levels are far below what Welch on feasible
record lengths resolves, so this metric is analytic-first (the pllsim
NoisePath budgets), with the time-domain PSD only as a near-out sanity
check (pllsim convention: S_phi is double-sideband, L(f) = S_phi/2).
"""
from __future__ import annotations

import numpy as np

TWOPI = 2.0 * np.pi


def adpll_rxband(pm, offsets_hz=(45e6, 80e6, 120e6)) -> dict:
    """L(f) [dBc/Hz] of an ADPLLTwoPoint phase path at the offsets.

    Interpolates the analyze() total budget (log-log).  Offsets beyond
    fref/2 fold in the discrete-time model — flagged, since there the
    DCO's continuous-time phase noise (the err-shaped Leeson term)
    dominates in hardware and the model's fold-back is only indicative.
    """
    ana = pm.analyze()
    lf = np.log10(ana.f)
    ls = np.log10(np.maximum(ana.pn_breakdown["total"], 1e-30))
    out = {}
    fref = pm.pll.cfg.fref
    for off in offsets_hz:
        s = 10.0 ** np.interp(np.log10(off), lf, ls)
        out[off] = {"ldbc_hz": float(10 * np.log10(s / 2.0)),
                    "beyond_fnyq": bool(off > fref / 2)}
    return out


def dtc_rxband(cfg, fs_bb: float, offsets_hz=(45e6, 80e6, 120e6)) -> dict:
    """L(f) [dBc/Hz] of the open-loop DTC phase path: locked-LO Leeson
    plus the (flat or dither-shaped) phase-quantization floor and the
    white jitter floor, all far-out."""
    out = {}
    q = cfg.lsb_rad
    for off in offsets_hz:
        s = 0.0
        if cfg.lo_pn is not None:
            s += float(cfg.lo_pn.leeson("lo").psd(
                np.array([max(off, cfg.lo_loop_bw)]))[0])
        s_q = q * q / 12.0 * 2.0 / fs_bb          # DSB white
        if cfg.dither:
            s_q *= 4.0 * np.sin(np.pi * off / fs_bb) ** 2
        s += s_q
        if cfg.jitter_rms_s > 0.0:
            s += (TWOPI * cfg.fout * cfg.jitter_rms_s) ** 2 * 2.0 / fs_bb
        out[off] = {"ldbc_hz": float(10 * np.log10(s / 2.0))}
    return out
