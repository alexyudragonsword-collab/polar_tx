# Vendored from pll_simulator@d7be4712: src/pllsim/core/noise.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Noise source PSD models and output noise budgeting.

Every source exposes .psd(f) in its native unit^2/Hz; the architecture supplies
the noise transfer function (NTF, a FreqResponse) that converts it to output
phase, forming a NoisePath.  output_psd() then produces the per-source
breakdown in rad^2/Hz (double-sideband, see core.jitter for the convention).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .freqresp import FreqResponse
from .jitter import sphi_from_ldbc

KB = 1.380649e-23
T0 = 290.0


@dataclass
class NoiseSource:
    """Base class: white PSD of `level` unit^2/Hz."""

    name: str
    unit: str = "rad^2/Hz"
    level: float = 0.0

    def psd(self, f: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(f).shape, self.level)


@dataclass
class FlickerFloorPhase(NoiseSource):
    """Phase noise with flat floor and 1/f flicker corner.

    S_phi(f) = floor * (1 + fc/f)   [rad^2/Hz, DSB]

    Construct from a spot L(f) in dBc/Hz via from_spot().
    """

    fc: float = 0.0  # flicker corner [Hz]

    @classmethod
    def from_spot(cls, name: str, l_floor_dbchz: float, fc: float = 0.0) -> "FlickerFloorPhase":
        return cls(name=name, unit="rad^2/Hz", level=float(sphi_from_ldbc(l_floor_dbchz)), fc=fc)

    def psd(self, f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        return self.level * (1.0 + self.fc / f)


@dataclass
class LeesonOscillator(NoiseSource):
    """Oscillator phase noise: 1/f^3 + 1/f^2 + floor.

    S_phi(f) = k2/f^2 * (1 + f_1f3/f) + floor   [rad^2/Hz DSB]

    Built from a spot phase noise L(f_spot) dominated by the 1/f^2 region.
    """

    k2: float = 0.0        # rad^2*Hz  (S_phi = k2/f^2 in the 20 dB/dec region)
    f_1f3: float = 0.0     # 1/f^3 corner [Hz]
    floor: float = 0.0     # rad^2/Hz

    @classmethod
    def from_spot(cls, name: str, l_dbchz: float, f_offset: float,
                  f_1f3: float = 0.0, floor_dbchz: float = -170.0) -> "LeesonOscillator":
        """Spot L at f_offset assumed on the 1/f^2 asymptote."""
        s_spot = float(sphi_from_ldbc(l_dbchz))
        k2 = s_spot * f_offset**2
        return cls(name=name, unit="rad^2/Hz", k2=k2, f_1f3=f_1f3,
                   floor=float(sphi_from_ldbc(floor_dbchz)))

    def psd(self, f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        return self.k2 / f**2 * (1.0 + self.f_1f3 / f) + self.floor


@dataclass
class CurrentNoise(NoiseSource):
    """Current noise: thermal floor + flicker.  S_i(f) = i2 * (1 + fc/f)  [A^2/Hz].

    `duty` scales the PSD for sources active only a fraction of the period
    (e.g. charge-pump on-time): effective S = duty * S_on.
    """

    i2: float = 0.0
    fc: float = 0.0
    duty: float = 1.0

    def psd(self, f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        return self.duty * self.i2 * (1.0 + self.fc / f)


@dataclass
class ResistorNoise(NoiseSource):
    """Thermal voltage noise of resistance r_ohm: 4kTR [V^2/Hz]."""

    r_ohm: float = 0.0
    temp_k: float = T0

    def psd(self, f: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(f).shape, 4.0 * KB * self.temp_k * self.r_ohm)


@dataclass
class SampledKTC(NoiseSource):
    """Sampled kT/C noise, aliased into [0, fs/2]:  S_v = 2kT/(C*fs)  [V^2/Hz]."""

    c_farad: float = 1e-12
    fs: float = 100e6
    temp_k: float = T0

    def psd(self, f: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(f).shape, 2.0 * KB * self.temp_k / (self.c_farad * self.fs))


@dataclass
class QuantizationPhase(NoiseSource):
    """Flat quantization noise.

    Total power q^2/12 spread uniformly over the Nyquist band [0, fs/2] on our
    one-sided frequency grid: S = (q^2/12)/(fs/2) = q^2/(6*fs), so that
    ∫ S df over [0, fs/2] = q^2/12.
    """

    q: float = 0.0   # quantization step in the native unit (rad or s)
    fs: float = 100e6

    def psd(self, f: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(f).shape, self.q**2 / (6.0 * self.fs))


@dataclass
class ShapedQuantization(NoiseSource):
    """Noise-shaped quantization: S(f) = q^2/(6*fs) * |2 sin(pi f/fs)|^(2m)."""

    q: float = 0.0
    fs: float = 100e6
    order: int = 1   # shaping order m

    def psd(self, f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        return self.q**2 / (6.0 * self.fs) * np.abs(2.0 * np.sin(np.pi * f / self.fs)) ** (2 * self.order)


@dataclass
class DcoQuantPhase(NoiseSource):
    """DCO frequency-quantization noise converted to phase.

    S_df(f) = kdco^2/(6*fs) * |2 sin(pi f/fs)|^(2*order)   [Hz^2/Hz]
    S_phi(f) = S_df(f) / f^2                                [rad^2/Hz]
    order=0: plain quantization; order=m: m-th order dithered at fs.
    """

    kdco: float = 1e3      # Hz per LSB
    fs: float = 100e6      # dither/update rate
    order: int = 1

    def psd(self, f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        s_df = self.kdco**2 / (6.0 * self.fs) * \
            np.abs(2.0 * np.sin(np.pi * f / self.fs)) ** (2 * self.order)
        return s_df / f**2


@dataclass
class TabulatedPhase(NoiseSource):
    """Piecewise log-log interpolated S_phi from (f, L_dbc) pairs."""

    f_pts: tuple = field(default_factory=tuple)
    l_dbc_pts: tuple = field(default_factory=tuple)

    def psd(self, f: np.ndarray) -> np.ndarray:
        lf = np.log10(np.asarray(f, dtype=float))
        li = np.interp(lf, np.log10(np.asarray(self.f_pts, dtype=float)),
                       np.asarray(self.l_dbc_pts, dtype=float))
        return sphi_from_ldbc(li)


# ------------------------------------------------------------------ budgeting
@dataclass
class NoisePath:
    source: NoiseSource
    ntf: FreqResponse   # converts source unit -> output phase [rad]


def output_psd(paths: list[NoisePath], f: np.ndarray) -> dict[str, np.ndarray]:
    """Per-source output phase PSD [rad^2/Hz DSB] plus 'total'."""
    out: dict[str, np.ndarray] = {}
    total = np.zeros(np.asarray(f).shape)
    for p in paths:
        s = p.source.psd(f) * p.ntf.mag2()
        out[p.source.name] = s
        total = total + s
    out["total"] = total
    return out
