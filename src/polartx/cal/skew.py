"""AM/PM path delay-skew estimation and correction.

Polar TX EVM/ACLR collapses when the envelope and phase paths are not
time-aligned (different pipeline depths, DPA vs modulator group delay).
The estimator reuses the vendored padpd alignment stack — complex
cross-correlation for the integer part, parabolic + cross-spectrum
phase-slope refinement for the fraction (sub-0.01-sample class).
"""
from __future__ import annotations

import numpy as np

from ..chain import PolarResult
from ..vendor.padpd.data.align import align_delay


def estimate_env_skew(res: PolarResult, max_lag: int = 256) -> dict:
    """Estimate the envelope-path delay [s] from a chain result.

    Correlates the (AC-coupled) commanded envelope against the output
    envelope |y| — exactly what a TX self-calibration does with its
    envelope-detector observation path.  Positive skew_s = envelope late.
    """
    x = res.env_cmd - res.env_cmd.mean()
    y = np.abs(res.y)
    y = y - y.mean()
    _, _, info = align_delay(x.astype(complex), y.astype(complex),
                             max_lag=max_lag)
    lag = info["lag_total"]       # positive = output envelope late = skew > 0
    return {"skew_samples": float(lag), "skew_s": float(lag / res.fs),
            "gain": info["gain"]}


def corrected_chain_config(cfg, skew_est_s: float):
    """New ChainConfig with the estimated skew pre-compensated."""
    from dataclasses import replace
    return replace(cfg, env_skew_s=cfg.env_skew_s - skew_est_s)


def estimate_skew_by_acp(tx, wf, *, span_s: float = 6e-9, n_grid: int = 7,
                         noise: bool = False, seed: int = 0) -> dict:
    """Power-detector-style skew search: no waveform-domain observation.

    AM/PM skew is the dominant creator of adjacent-channel power in a
    polar TX, so a chip with only a band-power detector can calibrate it
    by sweeping a digital trial delay and minimizing ACP.  Grid search
    over +/-span_s, then a parabolic fit through the best three points
    gives sub-grid resolution.  Returns the estimated skew (= minus the
    optimal correction) and the search trace.
    """
    from dataclasses import replace

    from ..vendor.padpd.metrics import aclr

    trials = np.linspace(-span_s, span_s, n_grid)
    cost = np.empty(n_grid)
    cfg0 = tx.cfg
    try:
        for i, t in enumerate(trials):
            tx.cfg = replace(cfg0, env_skew_s=cfg0.env_skew_s + t)
            res = tx.run(wf, noise=noise, seed=seed)
            a = aclr(res.y, res.fs, wf.bw)
            cost[i] = 10 ** (a["lower_dbc"] / 10) + 10 ** (a["upper_dbc"] / 10)
    finally:
        tx.cfg = cfg0
    k = int(np.argmin(cost))
    if 0 < k < n_grid - 1:                     # parabolic refinement
        c = 10 * np.log10(cost[k - 1: k + 2])
        denom = c[0] - 2 * c[1] + c[2]
        frac = 0.5 * (c[0] - c[2]) / denom if abs(denom) > 1e-12 else 0.0
        t_opt = trials[k] + frac * (trials[1] - trials[0])
    else:
        t_opt = trials[k]
    return {"skew_s": float(-t_opt), "trials_s": trials,
            "acp_db": 10 * np.log10(cost)}
