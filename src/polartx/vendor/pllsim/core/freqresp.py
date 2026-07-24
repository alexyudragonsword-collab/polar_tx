# Vendored from pll_simulator@d7be4712: src/pllsim/core/freqresp.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Grid-evaluated frequency-response algebra for PLL loop analysis.

All transfer functions are represented as complex responses H(f) evaluated on
a shared (usually log-spaced) offset-frequency grid.  This avoids the numeric
pathologies of polynomial transfer functions when a 12 GHz carrier meets a
100 kHz loop bandwidth, and lets s-domain, z-domain and transcendental
elements (pure delay, ZOH) mix freely.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TWOPI = 2.0 * np.pi


def default_grid(f1: float = 1e2, f2: float = 1e9, points_per_decade: int = 60) -> np.ndarray:
    """Log-spaced offset-frequency grid [Hz]."""
    n = int(np.ceil(np.log10(f2 / f1) * points_per_decade)) + 1
    return np.logspace(np.log10(f1), np.log10(f2), n)


class FreqResponse:
    """Complex frequency response on a fixed grid with operator algebra."""

    __slots__ = ("f", "h")

    def __init__(self, f: np.ndarray, h: np.ndarray | complex):
        self.f = np.asarray(f, dtype=float)
        h = np.asarray(h, dtype=complex)
        if h.ndim == 0:
            h = np.full(self.f.shape, complex(h))
        if h.shape != self.f.shape:
            raise ValueError("f and h must have the same shape")
        self.h = h

    # ---------------------------------------------------------------- ctors
    @classmethod
    def constant(cls, f: np.ndarray, value: complex) -> "FreqResponse":
        return cls(f, np.full(np.asarray(f).shape, complex(value)))

    @classmethod
    def from_callable(cls, f: np.ndarray, func) -> "FreqResponse":
        s = 1j * TWOPI * np.asarray(f, dtype=float)
        return cls(f, func(s))

    @classmethod
    def integrator(cls, f: np.ndarray, gain: float = 1.0) -> "FreqResponse":
        """gain / s  with s = j2πf."""
        return cls(f, gain / (1j * TWOPI * np.asarray(f, dtype=float)))

    @classmethod
    def differentiator(cls, f: np.ndarray, gain: float = 1.0) -> "FreqResponse":
        return cls(f, gain * 1j * TWOPI * np.asarray(f, dtype=float))

    @classmethod
    def delay(cls, f: np.ndarray, td: float) -> "FreqResponse":
        return cls(f, np.exp(-1j * TWOPI * np.asarray(f, dtype=float) * td))

    @classmethod
    def zdomain(cls, f: np.ndarray, num_z, den_z, fs: float) -> "FreqResponse":
        """Evaluate a z-domain rational function at z = exp(j2πf/fs).

        num_z, den_z: coefficient sequences in z^-1 (i.e. b0 + b1 z^-1 + ...).
        """
        zinv = np.exp(-1j * TWOPI * np.asarray(f, dtype=float) / fs)
        num = np.zeros_like(zinv, dtype=complex)
        for k, b in enumerate(num_z):
            num += b * zinv**k
        den = np.zeros_like(zinv, dtype=complex)
        for k, a in enumerate(den_z):
            den += a * zinv**k
        return cls(f, num / den)

    @classmethod
    def zoh(cls, f: np.ndarray, fs: float) -> "FreqResponse":
        """Zero-order-hold response (1 - z^-1)/(s Ts): sinc magnitude, half-sample delay."""
        f = np.asarray(f, dtype=float)
        x = np.pi * f / fs
        return cls(f, np.sinc(f / fs) * np.exp(-1j * x))

    @classmethod
    def accumulator(cls, f: np.ndarray, fs: float, gain: float = 1.0) -> "FreqResponse":
        """gain / (1 - z^-1) at z = exp(j2πf/fs) — discrete-time integrator."""
        zinv = np.exp(-1j * TWOPI * np.asarray(f, dtype=float) / fs)
        return cls(f, gain / (1.0 - zinv))

    # ------------------------------------------------------------- algebra
    def _coerce(self, other) -> np.ndarray:
        if isinstance(other, FreqResponse):
            if other.f.shape != self.f.shape or not np.allclose(other.f, self.f):
                raise ValueError("FreqResponse grids differ")
            return other.h
        return np.asarray(other, dtype=complex)

    def __add__(self, other):
        return FreqResponse(self.f, self.h + self._coerce(other))

    __radd__ = __add__

    def __sub__(self, other):
        return FreqResponse(self.f, self.h - self._coerce(other))

    def __rsub__(self, other):
        return FreqResponse(self.f, self._coerce(other) - self.h)

    def __mul__(self, other):
        return FreqResponse(self.f, self.h * self._coerce(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return FreqResponse(self.f, self.h / self._coerce(other))

    def __rtruediv__(self, other):
        return FreqResponse(self.f, self._coerce(other) / self.h)

    def __pow__(self, n):
        return FreqResponse(self.f, self.h**n)

    def __neg__(self):
        return FreqResponse(self.f, -self.h)

    def feedback(self, other=None) -> "FreqResponse":
        """Closed loop self / (1 + self*other); unity feedback if other is None."""
        if other is None:
            return FreqResponse(self.f, self.h / (1.0 + self.h))
        oh = self._coerce(other)
        return FreqResponse(self.f, self.h / (1.0 + self.h * oh))

    # -------------------------------------------------------------- views
    def mag(self) -> np.ndarray:
        return np.abs(self.h)

    def mag2(self) -> np.ndarray:
        return np.abs(self.h) ** 2

    def db(self) -> np.ndarray:
        return 20.0 * np.log10(np.maximum(np.abs(self.h), 1e-300))

    def deg_unwrapped(self) -> np.ndarray:
        return np.degrees(np.unwrap(np.angle(self.h)))


@dataclass(frozen=True)
class LoopMetrics:
    """Loop stability / bandwidth summary."""

    f_ugb: float          # unity-gain (gain-crossover) frequency [Hz]
    pm_deg: float         # phase margin [deg]
    gm_db: float          # gain margin [dB] (inf if no phase crossover)
    f_3db: float          # closed-loop -3 dB bandwidth [Hz]
    peaking_db: float     # closed-loop peaking above DC gain [dB]
    n_crossings: int      # number of gain crossovers (should be 1)


def _log_interp_crossing(f: np.ndarray, y: np.ndarray, level: float) -> list[float]:
    """Frequencies where y crosses `level` (log-x linear interpolation)."""
    out = []
    d = y - level
    sign = np.sign(d)
    for i in range(len(f) - 1):
        if sign[i] == 0:
            out.append(f[i])
        elif sign[i] * sign[i + 1] < 0:
            lf1, lf2 = np.log10(f[i]), np.log10(f[i + 1])
            t = d[i] / (d[i] - d[i + 1])
            out.append(10 ** (lf1 + t * (lf2 - lf1)))
    return out


def loop_metrics(gol: FreqResponse) -> LoopMetrics:
    """Compute UGB, PM, GM, closed-loop bandwidth and peaking from open loop."""
    f = gol.f
    mag_db = gol.db()
    ph = gol.deg_unwrapped()

    crossings = _log_interp_crossing(f, mag_db, 0.0)
    n_cross = len(crossings)
    if n_cross == 0:
        f_ugb = float("nan")
        pm = float("nan")
    else:
        f_ugb = crossings[0]
        ph_at = np.interp(np.log10(f_ugb), np.log10(f), ph)
        pm = 180.0 + ph_at
        # normalize in case phase was unwrapped past ±360
        pm = ((pm + 180.0) % 360.0) - 180.0
        if pm < 0:
            pm += 360.0 if pm < -180.0 else 0.0

    # gain margin: first -180 deg crossing beyond DC
    gm = float("inf")
    ph_cross = _log_interp_crossing(f, ph, -180.0)
    for fc in ph_cross:
        if np.isnan(f_ugb) or fc > f_ugb * 0.999:
            m = np.interp(np.log10(fc), np.log10(f), mag_db)
            gm = -m
            break

    # closed loop (unity feedback, input-referred lowpass)
    hcl = gol.feedback()
    hdb = hcl.db()
    ref_db = hdb[0]
    peaking = float(np.max(hdb) - ref_db)
    # first downward crossing: z-domain responses are periodic in f, so the
    # last crossing would land near an alias image instead of the loop BW
    c3 = _log_interp_crossing(f, hdb, ref_db - 3.0)
    f3 = c3[0] if c3 else float("nan")
    return LoopMetrics(
        f_ugb=float(f_ugb),
        pm_deg=float(pm),
        gm_db=float(gm),
        f_3db=float(f3),
        peaking_db=peaking,
        n_crossings=n_cross,
    )
