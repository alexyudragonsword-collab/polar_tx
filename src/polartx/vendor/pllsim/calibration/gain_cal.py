# Vendored from pll_simulator@d7be4712: src/pllsim/calibration/gain_cal.py
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
"""
from __future__ import annotations

import numpy as np


class KdcoCal:
    """Open-loop two-point Kdco estimator, run as a pre-lock FCAL phase.

    States: repeat `rounds` times {+A for meas_n cycles, -A for meas_n}.
    Feed the measured frequency each cycle via step(); .done flips True when
    finished and .value holds the Kdco estimate [Hz/LSB].
    """

    def __init__(self, kdco_init: float, amp_lsb: float = 8.0,
                 meas_n: int = 1024, rounds: int = 4, settle: int = 64):
        self.value = float(kdco_init)
        self.amp = float(amp_lsb)
        self.meas_n = int(meas_n)
        self.rounds = int(rounds)
        self.settle = int(settle)
        self.n = 0
        self._acc = 0.0
        self._cnt = 0
        self._half_means: list[float] = []
        self._ests: list[float] = []
        self.done = False
        self.trace: list[float] = []

    @property
    def total_cycles(self) -> int:
        return 2 * self.rounds * self.meas_n

    @property
    def perturbation(self) -> float:
        """OTW offset [LSB] for the current cal cycle (0 when done)."""
        if self.done:
            return 0.0
        return self.amp if (self.n // self.meas_n) % 2 == 0 else -self.amp

    def step(self, f_meas: float) -> float:
        if self.done:
            return self.value
        if (self.n % self.meas_n) >= self.settle:
            self._acc += f_meas
            self._cnt += 1
        self.n += 1
        if self.n % self.meas_n == 0 and self._cnt > 0:
            self._half_means.append(self._acc / self._cnt)
            self._acc = 0.0
            self._cnt = 0
            if len(self._half_means) % 2 == 0:
                f_hi, f_lo = self._half_means[-2], self._half_means[-1]
                self._ests.append((f_hi - f_lo) / (2.0 * self.amp))
        if self.n >= self.total_cycles:
            if self._ests:
                self.value = float(np.mean(self._ests))
            self.done = True
        self.trace.append(self.value)
        return self.value


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


class TdcPeriodCal:
    """EMA of TDC codes-per-DCO-period; .value converts code -> UI."""

    def __init__(self, cpp_init: float, ema: float = 1e-3):
        self.value = float(cpp_init)       # codes per period estimate
        self._ema = ema
        self.trace: list[float] = []

    def step(self, cpp_meas: float) -> float:
        self.value += self._ema * (cpp_meas - self.value)
        self.trace.append(self.value)
        return self.value

    def code_to_ui(self, code: float) -> float:
        return code / self.value
