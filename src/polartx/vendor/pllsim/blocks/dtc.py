# Vendored from pll_simulator@931cfaf: src/pllsim/blocks/dtc.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Digital-to-time converter with gain error, INL and jitter.

The loop requests a delay in seconds; the DTC computes a code using its
*calibrated* gain (gain_corr, updated by LMS) and the physical device applies
its *true* gain (1 + gain_error) plus INL and random jitter.  A mid-range
static offset keeps codes positive for bipolar targets (MASH residuals); the
constant part is absorbed by the loop as a static phase offset.

The code and time laws are kernels (core.jit) shared with the engines'
compiled loops; the class is the object view with its own RNG.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.jit import kernel


@dataclass
class DTCConfig:
    t_res: float                   # nominal LSB [s]
    n_bits: int = 10
    inl_poly: tuple = ()           # polynomial in normalized code x∈[0,1] -> seconds
    inl_sin: tuple = ()            # (amp_s, cycles, phase_rad): amp*sin(2π*cycles*x+φ)
    jitter_rms_s: float = 0.0
    gain_error_residual: float = 0.01   # residual gain error assumed in analyze()

    def __post_init__(self):
        if not self.t_res > 0:
            raise ValueError(f"DTCConfig t_res must be positive, got {self.t_res}")
        if int(self.n_bits) < 1:
            raise ValueError(f"DTCConfig n_bits must be >= 1, got {self.n_bits}")
        if self.jitter_rms_s < 0:
            raise ValueError(f"DTCConfig jitter_rms_s cannot be negative, got {self.jitter_rms_s}")
        if not 0.0 <= self.gain_error_residual < 1.0:
            raise ValueError("DTCConfig gain_error_residual is a fraction in "
                             f"[0, 1), got {self.gain_error_residual}")
        if self.inl_sin and len(self.inl_sin) != 3:
            raise ValueError("DTCConfig inl_sin is (amplitude_s, cycles, phase_rad) "
                             f"or empty, got {self.inl_sin}")

    @property
    def range_s(self) -> float:
        return self.t_res * ((1 << self.n_bits) - 1)

    # ---- kernel view: the INL description as arrays/scalars a kernel takes
    def inl_arrays(self) -> tuple[np.ndarray, bool, float, float, float]:
        poly = np.asarray(self.inl_poly, dtype=float)
        if self.inl_sin:
            amp, cyc, ph = (float(v) for v in self.inl_sin)
            return poly, True, amp, cyc, ph
        return poly, False, 0.0, 0.0, 0.0


@kernel
def dtc_code(t_target: float, range_s: float, gain_corr: float, t_res: float,
             code_max: int) -> int:
    """The code the loop asks for, for a bipolar target delay."""
    t_req = t_target + range_s / 2.0
    code = int(round(t_req * gain_corr / t_res))
    return min(max(code, 0), code_max)


@kernel
def dtc_inl_s(code: int, code_max: int, inl_poly: np.ndarray, has_sin: bool,
              sin_amp: float, sin_cyc: float, sin_ph: float) -> float:
    """INL [s] at a code: polynomial (ascending coefficients, Horner from
    the top as np.polyval does) plus an optional sinusoidal term."""
    x = code / code_max
    t = 0.0
    n = inl_poly.shape[0]
    if n > 0:
        y = 0.0
        for i in range(n - 1, -1, -1):
            y = y * x + inl_poly[i]
        t += y
    if has_sin:
        t += sin_amp * math.sin(2.0 * math.pi * sin_cyc * x + sin_ph)
    return t


@kernel
def dtc_time(code: int, t_res: float, gain_error: float, inl: float) -> float:
    """Physical delay [s] the device produces for a code, before jitter."""
    return code * t_res * (1.0 + gain_error) + inl


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
        self._inl = cfg.inl_arrays()

    def inl_s(self, code: int) -> float:
        return float(dtc_inl_s(int(code), self.code_max, *self._inl))

    def delay(self, t_target: float) -> float:
        """Physical delay produced for a requested (bipolar) target delay."""
        c = self.cfg
        code = int(dtc_code(t_target, c.range_s, self.gain_corr, c.t_res,
                            self.code_max))
        self.last_code = code
        self.last_code_centered = code - self.code_max / 2.0
        t = float(dtc_time(code, c.t_res, self.gain_error, self.inl_s(code)))
        if self.noise_on and c.jitter_rms_s > 0:
            t += c.jitter_rms_s * self.rng.standard_normal()
        return t
