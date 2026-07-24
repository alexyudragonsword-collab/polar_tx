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

from .characteristics import amam_curve, ampm_curve
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
