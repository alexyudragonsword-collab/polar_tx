"""Digital PA: amplitude code + phase-modulated carrier -> RF output.

Behavioral model at complex baseband: the envelope code selects the
per-code amplitude (unit-cell array with mismatch, composed with the
AM-AM law) and adds the code-dependent AM-PM phase; the carrier phase
comes from the phase modulator.  All code tables are precomputed once so
the sample path is a vectorized table lookup.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .characteristics import amam_curve, ampm_curve, efficiency_curve
from .mismatch import code_amplitude_table, inl_dnl


@dataclass
class DPAConfig:
    n_bits: int = 10
    n_thermo: int = 7               # thermometer MSBs, rest binary LSBs
    sigma_cell: float = 0.0         # relative random unit mismatch
    gradient: float = 0.0           # systematic tilt across the thermo array
    amam: object = "ideal"          # "ideal" | ("rapp", p, drive) | ("lut", r_in, r_out)
    ampm_deg_poly: tuple = ()       # AM-PM [deg] polynomial in code/fullscale
    ampm_lut: tuple | None = None   # (r_in, deg) measured AM-PM, overrides poly
    eff: tuple | None = ("scpa", 0.67, 0.85)   # drain-efficiency law
    seed: int = 0

    @property
    def n_codes(self) -> int:
        return 1 << self.n_bits


class DPA:
    def __init__(self, cfg: DPAConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        raw = code_amplitude_table(cfg.n_bits, min(cfg.n_thermo, cfg.n_bits),
                                   cfg.sigma_cell, cfg.gradient, rng)
        r = raw / raw[-1]                       # normalized array amplitude
        self.amp_table = amam_curve(cfg.amam, r)
        if cfg.ampm_lut is not None:
            r_in, deg = cfg.ampm_lut
            self.phase_table = np.deg2rad(
                np.interp(r, np.asarray(r_in, float),
                          np.asarray(deg, float)))
        else:
            self.phase_table = ampm_curve(cfg.ampm_deg_poly, r)
        self._mismatch = inl_dnl(raw)

    # ------------------------------------------------------------- codes
    def encode(self, env_norm: np.ndarray) -> np.ndarray:
        """Normalized envelope [0,1] -> amplitude code (round + clip)."""
        c = np.rint(np.asarray(env_norm, float) * (self.cfg.n_codes - 1))
        return np.clip(c, 0, self.cfg.n_codes - 1).astype(np.int64)

    def __call__(self, code: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """Complex-baseband DPA output, full scale = 1."""
        code = np.asarray(code, dtype=np.int64)
        return self.amp_table[code] * np.exp(
            1j * (np.asarray(phase, float) + self.phase_table[code]))

    def inl_dnl(self) -> dict:
        """Array INL/DNL (mismatch only, before the AM-AM law)."""
        return self._mismatch

    # -------------------------------------------------------- efficiency
    def efficiency(self, code: np.ndarray) -> np.ndarray:
        """Instantaneous drain efficiency per code (cfg.eff law)."""
        if self.cfg.eff is None:
            raise ValueError("DPAConfig.eff is None")
        return efficiency_curve(self.cfg.eff,
                                self.amp_table[np.asarray(code, np.int64)])

    def average_efficiency(self, code: np.ndarray) -> dict:
        """Modulated average: eta_avg = sum(P_out) / sum(P_dc).

        P_out ∝ amp², P_dc = P_out/η(amp) evaluated per sample; the polar
        TX's headline advantage over a linear PA is exactly this number
        at OFDM backoff."""
        code = np.asarray(code, np.int64)
        amp = self.amp_table[code]
        p_out = amp * amp
        eta = efficiency_curve(self.cfg.eff, amp)
        on = eta > 0
        p_dc = np.zeros_like(p_out)
        p_dc[on] = p_out[on] / eta[on]
        tot_dc = p_dc.sum()
        return {"eta_avg": float(p_out.sum() / tot_dc) if tot_dc else 0.0,
                "p_out_norm": float(p_out.mean()),
                "backoff_db": float(-10 * np.log10(max(p_out.mean(), 1e-30)))}
