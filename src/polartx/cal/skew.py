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
