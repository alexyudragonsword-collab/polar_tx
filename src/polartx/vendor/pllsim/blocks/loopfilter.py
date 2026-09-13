# Vendored from pll_simulator@931cfaf: src/pllsim/blocks/loopfilter.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Passive loop-filter state-space model with exact discrete-time updates.

Topologies (charge-pump drives node 1, vctrl taken at node 1):
  2nd order:  C2 from node1 to gnd, series (R2 + C1) from node1 to gnd
  3rd order:  2nd order + series R3 to node2 with C3 to gnd (vctrl at node2)

States are capacitor voltages.  The CP current is treated as an impulse of
charge dq per reference cycle near lock (|t_on| << Tref); during acquisition a
pulse-width-aware two-segment update is available (update_pulse).

Exact discretization: x(t+T) = e^{AT} x(t) + (charge impulse enters as an
instantaneous state jump dq/C_eff distributed by the input vector).

The three per-cycle updates are kernels (core.jit) over the state vector and
the precomputed matrices, shared with the analog engines' compiled loops.
They are written element by element on purpose: ``ad @ x`` goes to BLAS,
whose fused multiply-add differs from the explicit sum in the last bit for
about half of all 2x2 products, and the compiled and interpreted paths have
to agree bit for bit.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from ..core.jit import kernel


@dataclass
class FilterDesign:
    """Component values; c3/r3 = 0 disables the third pole."""
    c1: float
    r2: float
    c2: float
    r3: float = 0.0
    c3: float = 0.0

    def __post_init__(self):
        if not self.c1 > 0 or not self.c2 > 0:
            raise ValueError(f"FilterDesign c1 and c2 must be positive, got c1={self.c1}, c2={self.c2}")
        if self.r2 < 0:
            raise ValueError(f"FilterDesign r2 cannot be negative, got {self.r2}")
        if (self.r3 > 0) != (self.c3 > 0) or self.r3 < 0 or self.c3 < 0:
            # order() reads "third pole present" as both positive; one of the
            # two set is a filter that reads as third-order and is not
            raise ValueError("FilterDesign r3 and c3 must both be positive (a "
                             f"third pole) or both zero, got r3={self.r3}, c3={self.c3}")

    @property
    def order(self) -> int:
        return 3 if (self.c3 > 0 and self.r3 > 0) else 2


@kernel
def cdiv(a: complex, b: complex) -> complex:
    """a / b in real arithmetic, CPython's algorithm (_Py_c_quot).

    Written out because the interpreted path reads the eigenvalues out of a
    numpy array, and numpy's complex division multiplies by a reciprocal
    where CPython and numba divide: one bit apart once in a few thousand
    cycles, which is enough to fail the bit-identity test.  Float ops on
    numpy scalars are IEEE-identical; complex division is the exception.
    """
    ar, ai, br, bi = a.real, a.imag, b.real, b.imag
    if abs(br) >= abs(bi):
        if br == 0.0:
            return complex(math.nan, math.nan)
        ratio = bi / br
        denom = br + bi * ratio
        return complex((ar + ai * ratio) / denom, (ai - ar * ratio) / denom)
    ratio = br / bi
    denom = br * ratio + bi
    return complex((ar * ratio + ai) / denom, (ai * ratio - ar) / denom)


@kernel
def cmag(z: complex) -> float:
    """|z| in real arithmetic, for a branch threshold.

    Not `abs(z)`: Cython lowers that to `cabs()`, and the NDK's clang -- unlike
    the host compiler, which only warns -- rejects the implicit declaration,
    so the Android APK's compiled half failed to build while everything else
    was green.  A plain sqrt is also the one form all three lowerings
    (CPython, numba, Cython) agree on: CPython's `abs(complex)` is libm
    `hypot`, `math.hypot` is CPython's own correctly-rounded version, and
    they differ in 48 of 100 000 draws.

    Only ever compared against a threshold here, so the last bit does not
    reach a result; the two branches it selects agree to 1e-16 where they
    meet (test_seg_integral_branches_agree_where_they_meet).  It would
    overflow above |z| ~ 1e154, which a passive RC filter's eigenvalues
    cannot reach.
    """
    return math.sqrt(z.real * z.real + z.imag * z.imag)


@kernel
def cexpm1(z: complex) -> complex:
    """expm1 of a complex number, computed the way numpy computes it."""
    s = math.sin(z.imag / 2.0)
    return complex(math.expm1(z.real) * math.cos(z.imag) - 2.0 * s * s,
                   math.exp(z.real) * math.sin(z.imag))


@kernel
def lf_impulse(x: np.ndarray, ad: np.ndarray, b: np.ndarray, dq: float,
               tmp: np.ndarray) -> None:
    """One tstep with the CP charge dq applied as an impulse at the start.

    Impulse response: state jump x += b * dq (b in 1/C units), then free
    evolution over tstep.  Valid when the CP on-time << tstep.
    """
    n = x.shape[0]
    for i in range(n):
        tmp[i] = x[i] + b[i] * dq
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += ad[i, j] * tmp[j]
        x[i] = s


@kernel
def lf_pulse(x: np.ndarray, ad: np.ndarray, b: np.ndarray, w: np.ndarray,
             v: np.ndarray, vinv: np.ndarray, vinv_b: np.ndarray, i_cp: float,
             t_on: float, tstep: float, tmp: np.ndarray, xe: np.ndarray) -> None:
    """One tstep with constant current i_cp for t_on, then free evolution.

    A is singular (the type-II integrator pole: total charge is conserved
    with zero input); the input integral is evaluated per eigenvalue with
    the d->0 limit handled explicitly.  For t_on << tstep the charge
    impulse approximation (precomputed e^{A T}) is exact to O(t_on/tau)
    and avoids the per-cycle eigen-reconstruction entirely.
    """
    t_on = min(abs(t_on), tstep)
    if t_on < 0.02 * tstep:
        lf_impulse(x, ad, b, i_cp * t_on, tmp)
        return
    # e^{A t_on} and gamma = int_0^t e^{As} ds b via the eigenbasis;
    # (e^{wt}-1)/w cancels catastrophically for the near-zero (type-II)
    # eigenvalue, so use the Taylor branch there
    n = x.shape[0]
    rest = tstep - t_on
    for k in range(n):
        wt = w[k] * t_on
        ew = cmath.exp(wt)
        if cmag(wt) < 1e-8:
            g = t_on * (1.0 + wt / 2.0 + cdiv(wt * wt, complex(6.0, 0.0)))
        else:
            g = cdiv(ew - 1.0, w[k])
        s = 0j
        for j in range(n):
            s += vinv[k, j] * x[j]
        val = ew * s + g * vinv_b[k] * i_cp
        if rest > 0:
            val = cmath.exp(w[k] * rest) * val
        xe[k] = val
    for i in range(n):
        s = 0j
        for j in range(n):
            s += v[i, j] * xe[j]
        x[i] = s.real


@kernel
def seg_integral(t: float, a: float, b: float, wk: complex, tstep: float) -> complex:
    """(e^{w(t-a)+} - e^{w(t-b)+}) / w for one eigenvalue.

    The convolution of e^{At} with a unit current over [a, b], written so
    that both exponents are of *clipped elapsed* time and therefore never
    positive: the algebraically equivalent e^{wt}(e^{-wa} - e^{-wb})/w
    overflows for a fast pole, since e^{-w*tstep} is the reciprocal of a
    number the stable dynamics have already made tiny.  Clipping at zero
    also makes the term vanish identically before the segment starts.

    The type-II integrator sits exactly at w = 0, where the expression is
    0/0; that branch is the elapsed duration itself, plus one correction
    term for the case where |w*t| is merely small rather than zero.  One
    term is enough and the next would be unreachable: at the 1e-8 switch
    point the linear correction is already only ~6e-9 of the answer, so
    the quadratic one lands at ~1e-17 -- under the double-precision floor,
    where no test could tell it from nothing.
    """
    p = max(t - a, 0.0)
    q = max(t - b, 0.0)
    if cmag(wk) * tstep < 1e-8:
        return (p - q) + 0.5 * wk * (p * p - q * q)
    # expm1 rather than exp: for a slow pole the two exponentials are both
    # near 1 and their difference is all cancellation
    return cdiv(cexpm1(wk * p) - cexpm1(wk * q), wk)


@kernel
def lf_drive_fine(x: np.ndarray, w: np.ndarray, v: np.ndarray, vinv: np.ndarray,
                  vinv_b: np.ndarray, tstep: float, seg_amp: np.ndarray,
                  seg_dur: np.ndarray, n_seg: int, m: int, i_bias: float,
                  dq_impulse: float, out: np.ndarray, xe0: np.ndarray,
                  xe: np.ndarray) -> None:
    """One tstep driven by a piecewise-constant current, sampled m times.

    ``seg_amp``/``seg_dur`` hold ``n_seg`` (amplitude [A], duration [s])
    segments applied back to back from the start of the step; ``i_bias``
    (leakage) flows for the whole step and ``dq_impulse`` (noise charge)
    lands at t=0.  Samples are written to ``out`` (length m) at the END of
    each of the m equal sub-intervals, so the last one is the state the
    step leaves behind.

    Sampling inside the step is what makes the reference spur visible: the
    control node's ripple lives entirely within one reference period, so a
    record taken once per reference edge sees one point on it and reports no
    ripple at all.  The segments matter for the same reason -- see
    ChargePump.segments.

    Evaluated in closed form at all m sample times rather than by stepping.
    The system is LTI and diagonal in the eigenbasis, so each segment's
    contribution at time t is a difference of two exponentials of *elapsed*
    time -- see seg_integral.
    """
    # clip the drive to one step and build sub-interval boundaries
    amp2 = np.empty(n_seg + 1)
    dur2 = np.empty(n_seg + 1)
    used = 0.0
    ns = 0
    for sg in range(n_seg):
        dur = min(max(seg_dur[sg], 0.0), tstep - used)
        if dur <= 0.0:
            break
        amp2[ns] = seg_amp[sg] + i_bias
        dur2[ns] = dur
        ns += 1
        used += dur
    if used < tstep:
        amp2[ns] = i_bias
        dur2[ns] = tstep - used
        ns += 1
    n = x.shape[0]
    for k in range(n):
        s = 0j
        for j in range(n):
            s += vinv[k, j] * x[j]
        xe0[k] = s + vinv_b[k] * dq_impulse
    for si in range(m):
        t = tstep * (si + 1) / m
        for k in range(n):
            acc = 0j
            start = 0.0
            for sg in range(ns):
                a = amp2[sg]
                if a != 0.0:
                    acc += a * seg_integral(t, start, start + dur2[sg], w[k], tstep)
                start += dur2[sg]
            xe[k] = cmath.exp(t * w[k]) * xe0[k] + vinv_b[k] * acc
        s = 0j
        for j in range(n):
            s += v[n - 1, j] * xe[j]
        out[si] = s.real
    for i in range(n):
        s = 0j
        for j in range(n):
            s += v[i, j] * xe[j]
        x[i] = s.real


class LoopFilter:
    """Exact discrete state-space integrator for the passive CP filter."""

    def __init__(self, design: FilterDesign, tstep: float):
        self.d = design
        self.tstep = tstep
        self._build()

    def _build(self):
        d = self.d
        if d.order == 2:
            # states: v1 (C1 voltage), v2 (C2 = node voltage)
            # node eq: i_in = C2 dv2/dt + (v2 - v1)/R2 ; C1 dv1/dt = (v2 - v1)/R2
            a = np.array([
                [-1.0 / (d.r2 * d.c1), 1.0 / (d.r2 * d.c1)],
                [1.0 / (d.r2 * d.c2), -1.0 / (d.r2 * d.c2)],
            ])
            b = np.array([0.0, 1.0 / d.c2])       # per amp of input current
            c = np.array([0.0, 1.0])              # vctrl = v2 (node voltage)
        else:
            # states: v1 (C1), v2 (C2 node), v3 (C3 node = vctrl)
            a = np.array([
                [-1.0 / (d.r2 * d.c1), 1.0 / (d.r2 * d.c1), 0.0],
                [1.0 / (d.r2 * d.c2), -1.0 / d.c2 * (1.0 / d.r2 + 1.0 / d.r3), 1.0 / (d.r3 * d.c2)],
                [0.0, 1.0 / (d.r3 * d.c3), -1.0 / (d.r3 * d.c3)],
            ])
            b = np.array([0.0, 1.0 / d.c2, 0.0])
            c = np.array([0.0, 0.0, 1.0])
        self.a, self.b, self.c = a, b, c
        self.ad = expm(a * self.tstep)
        n = a.shape[0]
        self.x = np.zeros(n)
        # eigendecomposition for cheap per-cycle e^{At}: A = V diag(w) V^-1
        # (passive RC: distinct eigenvalues incl. the type-II zero eigenvalue);
        # kept complex whatever eig returns, so the kernels see one dtype
        w, v = np.linalg.eig(a)
        self._w = np.asarray(w, dtype=complex)
        self._v = np.asarray(v, dtype=complex)
        self._vinv = np.asarray(np.linalg.inv(v), dtype=complex)
        self._vinv_b = np.asarray(self._vinv @ b, dtype=complex)
        # scratch for the kernels
        self._tmp = np.empty(n)
        self._xe = np.empty(n, dtype=complex)
        self._xe0 = np.empty(n, dtype=complex)

    def reset(self, vctrl: float = 0.0):
        self.x[:] = vctrl

    @property
    def vctrl(self) -> float:
        # both topologies take vctrl at the last state (C @ x == x[-1])
        return float(self.x[-1])

    def update_impulse(self, dq: float) -> float:
        """One tstep with the CP charge dq applied as an impulse at the start."""
        lf_impulse(self.x, self.ad, self.b, float(dq), self._tmp)
        return self.vctrl

    def update_pulse(self, i_cp: float, t_on: float) -> float:
        """One tstep with constant current i_cp for t_on, then free evolution."""
        lf_pulse(self.x, self.ad, self.b, self._w, self._v, self._vinv,
                 self._vinv_b, float(i_cp), float(t_on), self.tstep, self._tmp,
                 self._xe)
        return self.vctrl

    def drive_fine(self, segments, m: int, i_bias: float = 0.0,
                   dq_impulse: float = 0.0) -> np.ndarray:
        """One tstep driven by a piecewise-constant current, sampled m times;
        see lf_drive_fine."""
        m = max(int(m), 1)
        segs = list(segments)
        amp = np.array([float(a) for a, _ in segs], dtype=float)
        dur = np.array([float(d) for _, d in segs], dtype=float)
        out = np.empty(m)
        lf_drive_fine(self.x, self._w, self._v, self._vinv, self._vinv_b,
                      self.tstep, amp, dur, len(segs), m, float(i_bias),
                      float(dq_impulse), out, self._xe0, self._xe)
        return out

    def _seg_integral(self, t: np.ndarray, a: float, b: float) -> np.ndarray:
        """seg_integral over a vector of sample times, one column per
        eigenvalue -- the array form the branch-agreement test drives."""
        t = np.asarray(t, dtype=float)
        out = np.empty((t.size, self._w.size), dtype=complex)
        for i, ti in enumerate(t):
            for k, wk in enumerate(self._w):
                out[i, k] = seg_integral(float(ti), float(a), float(b), complex(wk),
                                         self.tstep)
        return out

    # ------------------------------------------------------------ freq domain
    def transimpedance(self, f: np.ndarray) -> np.ndarray:
        """Z(f) = C (jwI - A)^-1 B  [V/A], vectorized over the grid."""
        w = 2j * np.pi * np.asarray(f, dtype=float)
        n = self.a.shape[0]
        z = np.empty(w.shape, dtype=complex)
        eye = np.eye(n)
        for i, wi in enumerate(w):
            z[i] = self.c @ np.linalg.solve(wi * eye - self.a, self.b)
        return z

    def charge_tf_z(self, f: np.ndarray) -> np.ndarray:
        """Exact discrete charge -> vctrl transfer [V/C] at z = e^{j2πf·tstep}.

        Matches update_impulse exactly: x[n] = Ad (x[n-1] + b dq[n])
        => H(z) = C (I - Ad z^-1)^-1 Ad b.  Use this for loops with aggressive
        BW/fref ratios where the continuous 1/s approximation misses the
        sampled-loop rolloff.
        """
        zinv = np.exp(-2j * np.pi * np.asarray(f, dtype=float) * self.tstep)
        n = self.a.shape[0]
        adb = self.ad @ self.b
        eye = np.eye(n)
        out = np.empty(zinv.shape, dtype=complex)
        for i, zi in enumerate(zinv):
            out[i] = self.c @ np.linalg.solve(eye - self.ad * zi, adb)
        return out
