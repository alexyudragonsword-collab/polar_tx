# Vendored from pll_simulator@d7be4712: src/pllsim/blocks/tdc.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Time-to-digital converter models: flash TDC and bang-bang PD."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TDCConfig:
    t_res: float                  # LSB [s]
    n_bits: int = 7               # range = 2^bits * t_res (should cover ~1 Tdco)
    inl_sin: tuple = ()           # (amp_s, cycles, phase) over the code range
    jitter_rms_s: float = 0.0     # input-referred random jitter
    gain_error: float = 0.0       # true LSB = t_res*(1+gain_error), unknown to loop


class TDC:
    """Flash TDC measuring a time interval in [0, range)."""

    def __init__(self, cfg: TDCConfig, rng: np.random.Generator, noise: bool = True):
        self.cfg = cfg
        self.rng = rng
        self.noise_on = noise
        self.code_max = (1 << cfg.n_bits) - 1
        self.t_lsb_true = cfg.t_res * (1.0 + cfg.gain_error)

    def measure(self, dt: float) -> int:
        """Quantize dt >= 0 to a code with the *true* (unknown) LSB + INL."""
        if self.noise_on and self.cfg.jitter_rms_s > 0:
            dt = dt + self.rng.normal(0.0, self.cfg.jitter_rms_s)
        code = int(dt / self.t_lsb_true)
        if self.cfg.inl_sin:
            amp, cyc, ph = self.cfg.inl_sin
            x = min(code, self.code_max) / self.code_max
            dt_inl = amp * np.sin(2 * np.pi * cyc * x + ph)
            code = int((dt + dt_inl) / self.t_lsb_true)
        return min(max(code, 0), self.code_max)


class BBPD:
    """Bang-bang phase detector: sign of the timing error plus jitter."""

    def __init__(self, jitter_rms_s: float, rng: np.random.Generator,
                 noise: bool = True):
        self.jitter = jitter_rms_s
        self.rng = rng
        self.noise_on = noise

    def sample(self, dt: float) -> int:
        if self.noise_on and self.jitter > 0:
            dt = dt + self.rng.normal(0.0, self.jitter)
        return 1 if dt >= 0 else -1
