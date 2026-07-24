"""Two-point (direct-path) gain calibration — offline estimator.

A direct-path gain error eps leaves the residual eps * highpass(phase)
at the modulator output.  Correlating the observed phase error against
the highpass-filtered command (both known to the DSP / feedback
receiver) gives eps in one least-squares shot; dividing dp_gain by
(1 + eps) matches the two points.  The in-loop sign-sign LMS variant
(Markulic-style background cal) is an M3 item — it needs a per-cycle
hook inside the event engine.
"""
from __future__ import annotations

import numpy as np


def estimate_dp_gain_error(pm, phase_cmd: np.ndarray,
                           phase_out: np.ndarray, fs_bb: float) -> dict:
    """LS estimate of the effective direct-path gain error.

    pm is an ADPLLTwoPoint (provides the loop's exact highpass 1 - h);
    phase_out the measured modulator output aligned with phase_cmd.
    """
    n = phase_cmd.size
    k = np.arange(n)
    ramp = phase_cmd[0] + (phase_cmd[-1] - phase_cmd[0]) * k / max(n - 1, 1)
    spec = np.fft.rfft(phase_cmd - ramp)
    f = np.fft.rfftfreq(n, 1.0 / fs_bb)
    c = pm.pll.cfg
    if c.mode == "tdc":
        gol, _ = pm.pll._gol_tdc(f[1:])
    else:
        gol, _, _ = pm.pll._gol_bbpd(f[1:])
    err = np.zeros(f.size, dtype=complex)   # highpass: err(0) = 1-h(0) = 0
    err[1:] = 1.0 - gol.feedback().h
    phi_hp = np.fft.irfft(spec * err, n=n)
    phi_hp = phi_hp - phi_hp.mean()

    resid = phase_out - phase_cmd
    resid = resid - np.polyval(np.polyfit(k, resid, 1), k)
    eps = float(np.dot(resid, phi_hp) / np.dot(phi_hp, phi_hp))
    return {"eps_hat": eps, "dp_gain_corr": 1.0 / (1.0 + eps),
            "hp_rms_rad": float(np.std(phi_hp))}
