# Vendored from pll_simulator@d7be4712: src/pllsim/core/deltasigma.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Delta-sigma modulators for fractional-N division and DCO dithering.

All modulators consume a fractional word `frac` in [0, 2^bits) and emit an
integer sequence whose mean is frac/2^bits.  `residual_ui()` returns the exact
accumulated quantization error

    e[n] = sum_k(y[k]) - sum_k(frac[k])/2^bits     [UI]

computed with integer arithmetic (bit-true, no float drift).  This is the
instantaneous divider phase error in units of one output period — the value a
DTC must cancel and the regressor that LMS gain calibration correlates with.
Noise shaping keeps e[n] bounded even though both running sums grow.
"""
from __future__ import annotations

import numpy as np


class _DsmBase:
    def __init__(self, bits: int = 24):
        self.bits = bits
        self.mod = 1 << bits
        self._sum_y = 0      # integer sum of outputs
        self._sum_frac = 0   # integer sum of frac words

    def _track(self, y: int, frac: int) -> int:
        self._sum_y += y
        self._sum_frac += frac
        return y

    def residual_ui(self) -> float:
        """Exact accumulated quantization error in UI (bounded by shaping)."""
        return float(self._sum_y * self.mod - self._sum_frac) / self.mod


class Efm1(_DsmBase):
    """First-order error-feedback modulator (plain accumulator), y in {0, 1}."""

    def __init__(self, bits: int = 24):
        super().__init__(bits)
        self.acc = 0

    def step(self, frac: int) -> int:
        self.acc += frac
        carry = self.acc >> self.bits
        self.acc &= self.mod - 1
        return self._track(int(carry), frac)


class Mash11(_DsmBase):
    """MASH 1-1: y in {-1, 0, 1, 2}, 2nd-order noise shaping."""

    def __init__(self, bits: int = 24):
        super().__init__(bits)
        self.a1 = 0
        self.a2 = 0
        self._c2_prev = 0

    def step(self, frac: int) -> int:
        self.a1 += frac
        c1 = self.a1 >> self.bits
        self.a1 &= self.mod - 1
        self.a2 += self.a1
        c2 = self.a2 >> self.bits
        self.a2 &= self.mod - 1
        out = c1 + c2 - self._c2_prev
        self._c2_prev = c2
        return self._track(int(out), frac)


class Mash111(_DsmBase):
    """MASH 1-1-1: y in {-3..4}, 3rd-order noise shaping."""

    def __init__(self, bits: int = 24):
        super().__init__(bits)
        self.a1 = 0
        self.a2 = 0
        self.a3 = 0
        self._c2p = 0
        self._c3p = 0
        self._c3pp = 0

    def step(self, frac: int) -> int:
        self.a1 += frac
        c1 = self.a1 >> self.bits
        self.a1 &= self.mod - 1
        self.a2 += self.a1
        c2 = self.a2 >> self.bits
        self.a2 &= self.mod - 1
        self.a3 += self.a2
        c3 = self.a3 >> self.bits
        self.a3 &= self.mod - 1
        # y = c1 + (1 - z^-1) c2 + (1 - z^-1)^2 c3
        out = c1 + (c2 - self._c2p) + (c3 - 2 * self._c3p + self._c3pp)
        self._c2p = c2
        self._c3pp = self._c3p
        self._c3p = c3
        return self._track(int(out), frac)


def run_sequence(mod, frac: int, n: int) -> np.ndarray:
    """Convenience: n modulator outputs for a constant frac word."""
    return np.array([mod.step(frac) for _ in range(n)], dtype=int)
