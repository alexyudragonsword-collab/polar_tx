# Vendored from pll_simulator@931cfaf: src/pllsim/calibration/gain_cal.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""DCO-gain and TDC-gain calibration.

KdcoCal — open-loop two-point FCAL (the way real ADPLLs sequence it): before
closing the loop, the OTW is stepped +A / -A for meas_n cycles each and the
average frequency of each half is measured from the counter (phase slope).
Kdco = dF/(2A).  A closed-loop two-point measurement would be useless: the
loop integrator absorbs the perturbation exactly, erasing the gain
information from the steady state.

TdcPeriodCal — Staszewski period normalization: the TDC measures the DCO
period every cycle; the running average of codes-per-period estimates
T_dco/t_res_true, which the loop uses to convert codes to UI.  TDC gain error
then cancels exactly in the code/codes-per-period ratio.

The per-cycle updates are kernels (core.jit) over small state vectors, shared
with the ADPLL's compiled loop; the classes are the object view.
"""
from __future__ import annotations

import numpy as np

from ..core.jit import kernel

# KdcoCal state vector (float64): cycle count, frequency accumulator, samples
# in the accumulator, done flag, estimate, half-means filled, estimates filled
KD_N, KD_ACC, KD_CNT, KD_DONE, KD_VALUE, KD_NHALF, KD_NEST = range(7)


@kernel
def kdco_perturbation(st: np.ndarray, amp: float, meas_n: int) -> float:
    """OTW offset [LSB] for the current cal cycle (0 when done)."""
    if st[KD_DONE] != 0.0:
        return 0.0
    return amp if (int(st[KD_N]) // meas_n) % 2 == 0 else -amp


@kernel
def kdco_step(st: np.ndarray, halves: np.ndarray, ests: np.ndarray, f_meas: float,
              amp: float, meas_n: int, rounds: int, settle: int) -> float:
    if st[KD_DONE] != 0.0:
        return st[KD_VALUE]
    n = int(st[KD_N])
    if (n % meas_n) >= settle:
        st[KD_ACC] += f_meas
        st[KD_CNT] += 1.0
    n += 1
    st[KD_N] = n
    if n % meas_n == 0 and st[KD_CNT] > 0.0:
        nh = int(st[KD_NHALF])
        halves[nh] = st[KD_ACC] / st[KD_CNT]
        nh += 1
        st[KD_NHALF] = nh
        st[KD_ACC] = 0.0
        st[KD_CNT] = 0.0
        if nh % 2 == 0:
            f_hi = halves[nh - 2]
            f_lo = halves[nh - 1]
            ne = int(st[KD_NEST])
            ests[ne] = (f_hi - f_lo) / (2.0 * amp)
            st[KD_NEST] = ne + 1
    if n >= 2 * rounds * meas_n:
        ne = int(st[KD_NEST])
        if ne > 0:
            s = 0.0
            for i in range(ne):
                s += ests[i]
            st[KD_VALUE] = s / ne
        st[KD_DONE] = 1.0
    return st[KD_VALUE]


class KdcoCal:
    """Open-loop two-point Kdco estimator, run as a pre-lock FCAL phase.

    States: repeat `rounds` times {+A for meas_n cycles, -A for meas_n}.
    Feed the measured frequency each cycle via step(); .done flips True when
    finished and .value holds the Kdco estimate [Hz/LSB].
    """

    def __init__(self, kdco_init: float, amp_lsb: float = 8.0,
                 meas_n: int = 1024, rounds: int = 4, settle: int = 64):
        self.amp = float(amp_lsb)
        self.meas_n = int(meas_n)
        self.rounds = int(rounds)
        self.settle = int(settle)
        self.st = np.zeros(7)
        self.st[KD_VALUE] = float(kdco_init)
        self.halves = np.zeros(2 * self.rounds)
        self.ests = np.zeros(self.rounds)
        self.trace: list[float] = []

    @property
    def value(self) -> float:
        return float(self.st[KD_VALUE])

    @property
    def n(self) -> int:
        return int(self.st[KD_N])

    @property
    def done(self) -> bool:
        return bool(self.st[KD_DONE] != 0.0)

    @property
    def total_cycles(self) -> int:
        return 2 * self.rounds * self.meas_n

    @property
    def perturbation(self) -> float:
        """OTW offset [LSB] for the current cal cycle (0 when done)."""
        return float(kdco_perturbation(self.st, self.amp, self.meas_n))

    def step(self, f_meas: float) -> float:
        v = float(kdco_step(self.st, self.halves, self.ests, float(f_meas),
                            self.amp, self.meas_n, self.rounds, self.settle))
        self.trace.append(v)
        return v


class BandSelect:
    """Binary-search coarse-band selection (pre-lock, open loop).

    Assumes band index -> frequency is monotonic (band pitch band_step_hz).
    Each trial band is measured for meas_n reference cycles by the counter
    (accuracy ~ fref/meas_n); ~log2(n_bands) trials.
    """

    def __init__(self, n_bands: int, f_target: float, meas_n: int = 64):
        self.f_target = f_target
        self.meas_n = meas_n
        self.lo, self.hi = 0, n_bands - 1
        self.band = (self.lo + self.hi) // 2
        self.best_band = self.band
        self.best_err = float("inf")
        self.trace: list[int] = []
        self.done = n_bands <= 1

    def observe(self, f_meas: float) -> int:
        """Feed the measured frequency of the current trial band; returns the
        next band to try (or the final choice once .done)."""
        self.trace.append(self.band)
        err = f_meas - self.f_target
        if abs(err) < self.best_err:
            self.best_err = abs(err)
            self.best_band = self.band
        if err < 0:
            self.lo = self.band + 1
        else:
            self.hi = self.band - 1
        if self.lo > self.hi:
            self.done = True
            self.band = self.best_band
        else:
            self.band = (self.lo + self.hi) // 2
        return self.band


@kernel
def tdc_period_step(st: np.ndarray, cpp_meas: float, ema: float) -> float:
    st[0] += ema * (cpp_meas - st[0])
    return st[0]


class TdcPeriodCal:
    """EMA of TDC codes-per-DCO-period; .value converts code -> UI."""

    def __init__(self, cpp_init: float, ema: float = 1e-3):
        self.st = np.array([float(cpp_init)])   # codes per period estimate
        self._ema = ema
        self.trace: list[float] = []

    @property
    def value(self) -> float:
        return float(self.st[0])

    def step(self, cpp_meas: float) -> float:
        v = float(tdc_period_step(self.st, float(cpp_meas), float(self._ema)))
        self.trace.append(v)
        return v

    def code_to_ui(self, code: float) -> float:
        return code / self.value
