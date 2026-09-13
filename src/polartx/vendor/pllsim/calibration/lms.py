# Vendored from pll_simulator@931cfaf: src/pllsim/calibration/lms.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""LMS-family calibrators.

Common contract: step(err, regressor) updates .value (or .lut); .trace records
the state each step for convergence plots.  Regressors are auto-centered with
an exponential moving average so callers can pass raw DSM residuals or codes.

Convergence time constants (docstring math, asserted x3 in tests):
  sign-sign LMS: tau ~ dynamic_range / (mu * f_update) update cycles for the
  initial error to slew out, then steady-state wander ~ mu * sqrt(f_update).

The update rules are kernels (core.jit) over small state vectors, shared with
the engines' compiled loops; the classes are the object view.  An engine
copies a calibrator's state into a vector before its loop and back after it,
so the object still carries its value from one run to the next.
"""
from __future__ import annotations

import math

import numpy as np

from ..core.jit import kernel, sgn

# gain-calibrator state vector (float64): value, regressor mean, error mean,
# step count
CAL_VALUE, CAL_REG_MEAN, CAL_ERR_MEAN, CAL_N = range(4)
CAL_STATE = 4
LMS_FULL, LMS_SIGN_SIGN = 0, 1


@kernel
def lms_step(kind: int, cs: np.ndarray, err: float, reg: float, mu: float,
             mu_final: float, gear_shift_n: float, ema: float,
             center_err: bool) -> float:
    """One update of a gain calibrator; returns the new value.

    ``mu_final`` is NaN and ``gear_shift_n`` negative when there is no gear
    shift.
    """
    cs[CAL_REG_MEAN] += ema * (reg - cs[CAL_REG_MEAN])
    d = reg - cs[CAL_REG_MEAN]
    if center_err:
        # Remove the error DC: the loop's equilibrium error is generally NOT
        # zero (CP mismatch/leakage static phase offset), which would saturate
        # sign(err) and blind the correlator.
        cs[CAL_ERR_MEAN] += ema * (err - cs[CAL_ERR_MEAN])
        e = err - cs[CAL_ERR_MEAN]
    else:
        e = err
    mu_now = mu
    if gear_shift_n >= 0.0 and not math.isnan(mu_final) and cs[CAL_N] > gear_shift_n:
        mu_now = mu_final
    if kind == LMS_FULL:
        cs[CAL_VALUE] += mu_now * e * d
    else:
        cs[CAL_VALUE] += mu_now * sgn(e) * sgn(d)
    cs[CAL_N] += 1.0
    return cs[CAL_VALUE]


class _CalBase:
    kind = LMS_FULL

    def __init__(self, init: float, mu: float, gear_shift_n: int | None = None,
                 mu_final: float | None = None, ema: float = 1e-3,
                 center_err: bool = True):
        self.value = float(init)
        self.mu = float(mu)
        self.mu_final = mu_final
        self.gear_shift_n = gear_shift_n
        self.n = 0
        self._ema = ema
        self._reg_mean = 0.0
        self._err_mean = 0.0
        self.center_err = center_err
        self.trace: list[float] = []

    # ---- kernel view
    def params(self) -> tuple[float, float, float, float, bool]:
        """(mu, mu_final or NaN, gear_shift_n or -1, ema, center_err)."""
        return (self.mu,
                float("nan") if self.mu_final is None else float(self.mu_final),
                -1.0 if self.gear_shift_n is None else float(self.gear_shift_n),
                float(self._ema), bool(self.center_err))

    def state(self) -> np.ndarray:
        return np.array([self.value, self._reg_mean, self._err_mean, float(self.n)])

    def load_state(self, cs: np.ndarray) -> None:
        self.value = float(cs[CAL_VALUE])
        self._reg_mean = float(cs[CAL_REG_MEAN])
        self._err_mean = float(cs[CAL_ERR_MEAN])
        self.n = int(cs[CAL_N])

    def step(self, err: float, reg: float) -> float:
        cs = self.state()
        v = float(lms_step(self.kind, cs, float(err), float(reg), *self.params()))
        self.load_state(cs)
        self.trace.append(v)
        return v


class LMSGainCal(_CalBase):
    """value += mu * centered(err) * centered(reg)   (full-precision LMS)."""
    kind = LMS_FULL


class SignSignLMS(_CalBase):
    """value += mu * sign(centered(err)) * sign(centered(reg))."""
    kind = LMS_SIGN_SIGN


# LUT calibrator scalar state: error mean, step count
LUT_ERR_MEAN, LUT_N = range(2)


@kernel
def lut_bin(reg: float, lo: float, hi: float, k: int) -> int:
    i = int((reg - lo) / (hi - lo) * k)
    return min(max(i, 0), k - 1)


@kernel
def lut_step(lut: np.ndarray, counts: np.ndarray, xs: np.ndarray, ls: np.ndarray,
             err: float, reg: float, lo: float, hi: float, k: int, mu: float,
             ema: float, ortho_every: int, snaps: np.ndarray, n_snap: int) -> int:
    """One LUT update; returns the snapshot count (it grows by one every
    ``ortho_every`` steps, when the mean and ramp are projected out)."""
    i = lut_bin(reg, lo, hi, k)
    ls[LUT_ERR_MEAN] += ema * (err - ls[LUT_ERR_MEAN])
    lut[i] += mu * sgn(err - ls[LUT_ERR_MEAN])
    counts[i] += 1.0
    ls[LUT_N] += 1.0
    if int(ls[LUT_N]) % ortho_every == 0:
        # visit-weighted mean/ramp projection restricted to visited bins:
        # unvisited bins are never written (they would accumulate projection
        # residue with no correcting updates) and stay at zero
        tot = 0.0
        for j in range(k):
            tot += counts[j]
        norm = max(tot, 1.0)
        mean = 0.0
        xmean = 0.0
        for j in range(k):
            w = counts[j] / norm
            mean += w * lut[j]
            xmean += w * xs[j]
        denom = 0.0
        num = 0.0
        for j in range(k):
            w = counts[j] / norm
            xw = xs[j] - xmean
            denom += w * (xw * xw)
            num += w * (lut[j] * xw)
        slope = num / denom if denom > 0.0 else 0.0
        for j in range(k):
            if counts[j] > 0.0:
                lut[j] -= mean + slope * (xs[j] - xmean)
        if n_snap < snaps.shape[0]:
            for j in range(k):
                snaps[n_snap, j] = lut[j]
        n_snap += 1
    return n_snap


class LUTCal:
    """Piecewise (K-segment) correction LUT over the regressor range [lo, hi].

    lut[k] += mu * sign(err) for the active bin — corrects DTC INL shape.
    Gain/offset components are left to a separate gain calibrator; to keep the
    LUT orthogonal to it, the mean and the linear ramp are projected out of
    the LUT after each update epoch (cheap: every `ortho_every` steps).
    """

    def __init__(self, k: int, lo: float, hi: float, mu: float,
                 ortho_every: int = 4096, ema: float = 1e-3):
        self.k = k
        self.lo, self.hi = lo, hi
        self.mu = mu
        self.lut = np.zeros(k)
        self.counts = np.zeros(k)
        self.ortho_every = ortho_every
        self.n = 0
        self._ema = ema
        self._err_mean = 0.0
        self.trace: list[np.ndarray] = []
        self._x = (np.arange(k) - (k - 1) / 2.0)

    # ---- kernel view
    def state(self) -> np.ndarray:
        return np.array([self._err_mean, float(self.n)])

    def load_state(self, ls: np.ndarray) -> None:
        self._err_mean = float(ls[LUT_ERR_MEAN])
        self.n = int(ls[LUT_N])

    def bin_index(self, reg: float) -> int:
        return int(lut_bin(float(reg), self.lo, self.hi, self.k))

    def correction(self, reg: float) -> float:
        return float(self.lut[self.bin_index(reg)])

    def step(self, err: float, reg: float) -> float:
        ls = self.state()
        snaps = np.empty((1, self.k))
        n = int(lut_step(self.lut, self.counts, self._x, ls, float(err), float(reg),
                         self.lo, self.hi, self.k, self.mu, self._ema,
                         self.ortho_every, snaps, 0))
        self.load_state(ls)
        if n:
            self.trace.append(snaps[0].copy())
        return float(self.lut[self.bin_index(reg)])
