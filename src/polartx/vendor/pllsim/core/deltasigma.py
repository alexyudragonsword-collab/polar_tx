# Vendored from pll_simulator@931cfaf: src/pllsim/core/deltasigma.py
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

The arithmetic lives in two kernels over an int64 state vector so the
engines' compiled loops and these classes run the same code; the classes are
the object view of that state (see core.jit for the rules).
"""
from __future__ import annotations

import numpy as np

from .jit import kernel

# state vector layout (int64): three accumulators, the delayed carries, and
# the two running sums residual_ui() is built from
A1, A2, A3, C2P, C3P, C3PP, SUM_Y, SUM_FRAC = range(8)
MASH_STATE = 8


@kernel
def mash_step(order: int, bits: int, st: np.ndarray, frac: int) -> int:
    """Advance a MASH of the given order one cycle; returns the integer output."""
    mask = (1 << bits) - 1
    st[A1] += frac
    c1 = st[A1] >> bits
    st[A1] &= mask
    if order == 1:
        out = c1
    else:
        st[A2] += st[A1]
        c2 = st[A2] >> bits
        st[A2] &= mask
        if order == 2:
            out = c1 + c2 - st[C2P]
            st[C2P] = c2
        else:
            st[A3] += st[A2]
            c3 = st[A3] >> bits
            st[A3] &= mask
            # y = c1 + (1 - z^-1) c2 + (1 - z^-1)^2 c3
            out = c1 + (c2 - st[C2P]) + (c3 - 2 * st[C3P] + st[C3PP])
            st[C2P] = c2
            st[C3PP] = st[C3P]
            st[C3P] = c3
    st[SUM_Y] += out
    st[SUM_FRAC] += frac
    return out


@kernel
def mash_residual(bits: int, st: np.ndarray) -> float:
    """Exact accumulated quantization error in UI (bounded by shaping)."""
    mod = 1 << bits
    return float(st[SUM_Y] * mod - st[SUM_FRAC]) / mod


def new_mash_state() -> np.ndarray:
    return np.zeros(MASH_STATE, dtype=np.int64)


class _DsmBase:
    order = 0

    def __init__(self, bits: int = 24):
        self.bits = bits
        self.mod = 1 << bits
        self.st = new_mash_state()

    def step(self, frac: int) -> int:
        """Advance one reference cycle."""
        return int(mash_step(self.order, self.bits, self.st, int(frac)))

    def residual_ui(self) -> float:
        """Exact accumulated quantization error in UI (bounded by shaping)."""
        return float(mash_residual(self.bits, self.st))

    # the golden-vector exporter reads the running sums directly
    @property
    def _sum_y(self) -> int:
        return int(self.st[SUM_Y])

    @property
    def _sum_frac(self) -> int:
        return int(self.st[SUM_FRAC])


class Efm1(_DsmBase):
    """First-order error-feedback modulator (plain accumulator), y in {0, 1}."""
    order = 1


class Mash11(_DsmBase):
    """MASH 1-1: y in {-1, 0, 1, 2}, 2nd-order noise shaping."""
    order = 2


class Mash111(_DsmBase):
    """MASH 1-1-1: y in {-3..4}, 3rd-order noise shaping."""
    order = 3


def run_sequence(mod, frac: int, n: int) -> np.ndarray:
    """Convenience: n modulator outputs for a constant frac word."""
    return np.array([mod.step(frac) for _ in range(n)], dtype=int)
