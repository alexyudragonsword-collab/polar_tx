# Vendored from pll_simulator@931cfaf: src/pllsim/blocks/tdc.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Time-to-digital converter models: flash TDC and bang-bang PD.

The quantizer and the decision are kernels (core.jit) shared with the ADPLL's
compiled loop; the classes are the object view with their own RNG.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.jit import kernel


@dataclass
class TDCConfig:
    t_res: float                  # LSB [s]
    n_bits: int = 7               # range = 2^bits * t_res (should cover ~1 Tdco)
    inl_sin: tuple = ()           # (amp_s, cycles, phase) over the code range
    jitter_rms_s: float = 0.0     # input-referred random jitter
    gain_error: float = 0.0       # true LSB = t_res*(1+gain_error), unknown to loop

    def __post_init__(self):
        if not self.t_res > 0:
            raise ValueError(f"TDCConfig t_res must be positive, got {self.t_res}")
        if int(self.n_bits) < 1:
            raise ValueError(f"TDCConfig n_bits must be >= 1, got {self.n_bits}")
        if self.jitter_rms_s < 0:
            raise ValueError(f"TDCConfig jitter_rms_s cannot be negative, got {self.jitter_rms_s}")
        if self.inl_sin and len(self.inl_sin) != 3:
            raise ValueError("TDCConfig inl_sin is (amplitude_s, cycles, phase_rad) "
                             f"or empty, got {self.inl_sin}")

    def inl_scalars(self) -> tuple[bool, float, float, float]:
        if self.inl_sin:
            amp, cyc, ph = (float(v) for v in self.inl_sin)
            return True, amp, cyc, ph
        return False, 0.0, 0.0, 0.0


@kernel
def tdc_measure(dt: float, t_lsb_true: float, code_max: int, has_sin: bool,
                sin_amp: float, sin_cyc: float, sin_ph: float) -> int:
    """Quantize dt >= 0 (jitter already applied) with the true LSB + INL."""
    code = int(dt / t_lsb_true)
    if has_sin:
        x = min(code, code_max) / code_max
        dt_inl = sin_amp * math.sin(2.0 * math.pi * sin_cyc * x + sin_ph)
        code = int((dt + dt_inl) / t_lsb_true)
    return min(max(code, 0), code_max)


@kernel
def bbpd_decide(dt: float) -> int:
    """The bang-bang decision for a timing error (jitter already applied)."""
    return 1 if dt >= 0 else -1


class TDC:
    """Flash TDC measuring a time interval in [0, range)."""

    def __init__(self, cfg: TDCConfig, rng: np.random.Generator, noise: bool = True):
        self.cfg = cfg
        self.rng = rng
        self.noise_on = noise
        self.code_max = (1 << cfg.n_bits) - 1
        self.t_lsb_true = cfg.t_res * (1.0 + cfg.gain_error)
        self._inl = cfg.inl_scalars()

    def measure(self, dt: float) -> int:
        """Quantize dt >= 0 to a code with the *true* (unknown) LSB + INL."""
        if self.noise_on and self.cfg.jitter_rms_s > 0:
            dt = dt + self.cfg.jitter_rms_s * self.rng.standard_normal()
        return int(tdc_measure(dt, self.t_lsb_true, self.code_max, *self._inl))


class BBPD:
    """Bang-bang phase detector: sign of the timing error plus jitter.

    ``meta_window_s`` is the resolution window of the sampling flop.  Inside it
    the flop cannot resolve in the time available, so the decision is a coin
    flip -- it still puts out +/-1, just not the right one.  That is a gain
    loss, not a noise gain: the output power stays at 1 either way, while the
    slope of the averaged characteristic drops (see meta_gain_penalty).

    The coin is the sign of a standard-normal draw, so the compiled loop can
    take it from the same pool as every other draw of the run (a uniform
    would need a second stream).  Runs with a metastability window therefore
    do not reproduce pre-0.9.5 realizations bit for bit; all others do.
    """

    def __init__(self, jitter_rms_s: float, rng: np.random.Generator,
                 noise: bool = True, meta_window_s: float = 0.0):
        self.jitter = jitter_rms_s
        self.rng = rng
        self.noise_on = noise
        self.meta_window_s = meta_window_s
        self.n_meta = 0

    def sample(self, dt: float) -> int:
        if self.noise_on and self.jitter > 0:
            dt = dt + self.jitter * self.rng.standard_normal()
        if self.meta_window_s > 0 and abs(dt) < self.meta_window_s:
            self.n_meta += 1
            if self.noise_on:
                return 1 if self.rng.standard_normal() < 0.0 else -1
        return int(bbpd_decide(dt))


def meta_gain_penalty(meta_window_s: float, sigma_t: float) -> float:
    """Factor on Kbb from a metastability window, in (0, 1].

    With input jitter sigma and a coin-flip window +/-W, the averaged
    characteristic is E[out|dt] = Q((W-dt)/sigma) - Q((W+dt)/sigma), whose
    slope at the origin is 2*phi(W/sigma)/sigma.  Dividing by the W=0 value
    sqrt(2/pi)/sigma leaves

        Kbb(W)/Kbb(0) = exp(-W^2 / (2 sigma^2)) .

    The output power is unchanged (a coin flip is still +/-1), so the whole
    effect lands on the gain: input-referred noise rises by exactly the
    reciprocal.  A window equal to sigma costs 4.34 dB.
    """
    if meta_window_s <= 0.0 or sigma_t <= 0.0:
        return 1.0
    return float(np.exp(-0.5 * (meta_window_s / sigma_t) ** 2))
