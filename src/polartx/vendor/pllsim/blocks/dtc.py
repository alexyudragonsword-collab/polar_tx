# Vendored from pll_simulator@d7be4712: src/pllsim/blocks/dtc.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Digital-to-time converter with gain error, INL and jitter.

The loop requests a delay in seconds; the DTC computes a code using its
*calibrated* gain (gain_corr, updated by LMS) and the physical device applies
its *true* gain (1 + gain_error) plus INL and random jitter.  A mid-range
static offset keeps codes positive for bipolar targets (MASH residuals); the
constant part is absorbed by the loop as a static phase offset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DTCConfig:
    t_res: float                   # nominal LSB [s]
    n_bits: int = 10
    inl_poly: tuple = ()           # polynomial in normalized code x∈[0,1] -> seconds
    inl_sin: tuple = ()            # (amp_s, cycles, phase_rad): amp*sin(2π*cycles*x+φ)
    jitter_rms_s: float = 0.0
    gain_error_residual: float = 0.01   # residual gain error assumed in analyze()

    @property
    def range_s(self) -> float:
        return self.t_res * ((1 << self.n_bits) - 1)


class DTC:
    def __init__(self, cfg: DTCConfig, rng: np.random.Generator,
                 noise: bool = True, gain_error: float = 0.0):
        self.cfg = cfg
        self.rng = rng
        self.noise_on = noise
        self.gain_error = gain_error   # true (unknown) fractional gain error
        self.gain_corr = 1.0           # calibrated code-domain correction
        self.code_max = (1 << cfg.n_bits) - 1
        self.last_code = 0
        self.last_code_centered = 0.0

    def inl_s(self, code: int) -> float:
        x = code / self.code_max
        t = 0.0
        if self.cfg.inl_poly:
            t += float(np.polyval(self.cfg.inl_poly[::-1], x))
        if self.cfg.inl_sin:
            amp, cyc, ph = self.cfg.inl_sin
            t += amp * np.sin(2.0 * np.pi * cyc * x + ph)
        return t

    def delay(self, t_target: float) -> float:
        """Physical delay produced for a requested (bipolar) target delay."""
        c = self.cfg
        t_req = t_target + c.range_s / 2.0
        code = int(round(t_req * self.gain_corr / c.t_res))
        code = min(max(code, 0), self.code_max)
        self.last_code = code
        self.last_code_centered = code - self.code_max / 2.0
        t = code * c.t_res * (1.0 + self.gain_error) + self.inl_s(code)
        if self.noise_on and c.jitter_rms_s > 0:
            t += self.rng.normal(0.0, c.jitter_rms_s)
        return t
