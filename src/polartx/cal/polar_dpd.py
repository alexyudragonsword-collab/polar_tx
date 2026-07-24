"""Polar DPD: the two-LUT predistorter of every digital polar TX.

AM-AM: an inverse-amplitude LUT re-maps the envelope command so the DPA
output amplitude is linear in the command.  AM-PM: the code-dependent
phase is subtracted from the phase command.  Built either from the known
DPA model (from_dpa - factory calibration) or from a chain measurement
(fit - binned |y| vs command, the envelope-detector observation path).
"""
from __future__ import annotations

import numpy as np


class PolarDPD:
    def __init__(self, amp_in: np.ndarray, amp_out: np.ndarray,
                 phase_corr_rad: np.ndarray):
        """amp_in -> amp_out: inverse AM-AM (both normalized [0,1]);
        phase_corr_rad: per-point AM-PM correction sampled on amp_in."""
        self.amp_in = np.asarray(amp_in, float)
        self.amp_out = np.asarray(amp_out, float)
        self.phase_corr = np.asarray(phase_corr_rad, float)

    # ------------------------------------------------------------ build
    @classmethod
    def from_dpa(cls, dpa, n: int = 1024) -> "PolarDPD":
        """Exact inversion of a known DPA's code tables."""
        code_norm = np.arange(dpa.cfg.n_codes) / (dpa.cfg.n_codes - 1)
        amp = dpa.amp_table / dpa.amp_table[-1]
        r = np.linspace(0.0, 1.0, n)
        # invert the (monotone) AM-AM: desired amp r -> command
        cmd = np.interp(r, amp, code_norm)
        ph = np.interp(cmd, code_norm, dpa.phase_table)
        return cls(r, cmd, ph)

    @classmethod
    def fit(cls, res, n_bins: int = 64) -> "PolarDPD":
        """From a chain measurement: bin output amplitude vs envelope
        command (and output-phase error vs command) - what a TX self-cal
        observes through its envelope detector / feedback receiver."""
        cmd = res.env_cmd
        out = np.abs(res.y)
        out = out / out.max()
        pherr = np.angle(res.y * np.exp(-1j * res.phase_out))
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(cmd, edges) - 1, 0, n_bins - 1)
        amp_o = np.full(n_bins, np.nan)
        ph_o = np.full(n_bins, np.nan)
        for b in range(n_bins):
            m = idx == b
            if m.sum() >= 4:
                amp_o[b] = out[m].mean()
                ph_o[b] = pherr[m].mean()
        centers = 0.5 * (edges[:-1] + edges[1:])
        ok = np.isfinite(amp_o)
        centers, amp_o, ph_o = centers[ok], amp_o[ok], ph_o[ok]
        amp_o = np.maximum.accumulate(amp_o)          # enforce monotone
        r = np.linspace(0.0, 1.0, 256)
        cmd_for_r = np.interp(r, amp_o / amp_o[-1], centers)
        ph_for_r = np.interp(cmd_for_r, centers, ph_o)
        return cls(r, cmd_for_r, ph_for_r)

    # ------------------------------------------------------------ apply
    def predistort(self, env_cmd: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
        """(predistorted envelope command, phase correction to subtract)."""
        e = np.clip(env_cmd, 0.0, 1.0)
        return (np.interp(e, self.amp_in, self.amp_out),
                np.interp(e, self.amp_in, self.phase_corr))
