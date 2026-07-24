# Vendored from pll_simulator@d7be4712: src/pllsim/calibration/lms.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""LMS-family calibrators.

Common contract: step(err, regressor) updates .value (or .lut); .trace records
the state each step for convergence plots.  Regressors are auto-centered with
an exponential moving average so callers can pass raw DSM residuals or codes.

Convergence time constants (docstring math, asserted x3 in tests):
  sign-sign LMS: tau ~ dynamic_range / (mu * f_update) update cycles for the
  initial error to slew out, then steady-state wander ~ mu * sqrt(f_update).
"""
from __future__ import annotations

import numpy as np


class _CalBase:
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

    def _mu_now(self) -> float:
        if self.gear_shift_n is not None and self.mu_final is not None \
                and self.n > self.gear_shift_n:
            return self.mu_final
        return self.mu

    def _center(self, reg: float) -> float:
        self._reg_mean += self._ema * (reg - self._reg_mean)
        return reg - self._reg_mean

    def _center_e(self, err: float) -> float:
        """Remove the error DC: the loop's equilibrium error is generally NOT
        zero (CP mismatch/leakage static phase offset), which would saturate
        sign(err) and blind the correlator."""
        if not self.center_err:
            return err
        self._err_mean += self._ema * (err - self._err_mean)
        return err - self._err_mean

    def step(self, err: float, reg: float) -> float:
        raise NotImplementedError


class LMSGainCal(_CalBase):
    """value += mu * centered(err) * centered(reg)   (full-precision LMS)."""

    def step(self, err: float, reg: float) -> float:
        d = self._center(reg)
        e = self._center_e(err)
        self.value += self._mu_now() * e * d
        self.n += 1
        self.trace.append(self.value)
        return self.value


class SignSignLMS(_CalBase):
    """value += mu * sign(centered(err)) * sign(centered(reg))."""

    def step(self, err: float, reg: float) -> float:
        d = self._center(reg)
        e = self._center_e(err)
        self.value += self._mu_now() * np.sign(e) * np.sign(d)
        self.n += 1
        self.trace.append(self.value)
        return self.value


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

    def bin_index(self, reg: float) -> int:
        i = int((reg - self.lo) / (self.hi - self.lo) * self.k)
        return min(max(i, 0), self.k - 1)

    def correction(self, reg: float) -> float:
        return float(self.lut[self.bin_index(reg)])

    def step(self, err: float, reg: float) -> float:
        i = self.bin_index(reg)
        self._err_mean += self._ema * (err - self._err_mean)
        self.lut[i] += self.mu * np.sign(err - self._err_mean)
        self.counts[i] += 1
        self.n += 1
        if self.n % self.ortho_every == 0:
            # visit-weighted mean/ramp projection restricted to visited bins:
            # unvisited bins are never written (they would accumulate projection
            # residue with no correcting updates) and stay at zero
            vis = self.counts > 0
            w = np.where(vis, self.counts, 0.0)
            w = w / max(w.sum(), 1.0)
            mean = float(w @ self.lut)
            xw = self._x - float(w @ self._x)
            denom = float(w @ (xw * xw))
            slope = float(w @ (self.lut * xw)) / denom if denom > 0 else 0.0
            self.lut[vis] -= mean + slope * xw[vis]
            self.trace.append(self.lut.copy())
        return float(self.lut[i])
