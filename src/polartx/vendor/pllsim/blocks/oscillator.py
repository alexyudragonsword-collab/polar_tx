# Vendored from pll_simulator@d7be4712: src/pllsim/blocks/oscillator.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Behavioral controlled oscillator (VCO/DCO).

Frequency law plus a Leeson noise profile.  Time-domain sims run at reference
rate; the per-step OscPhaseNoiseGen sample represents the oscillator phase
error accumulated over that interval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.colored import OscPhaseNoiseGen
from ..core.noise import LeesonOscillator


@dataclass
class OscConfig:
    f0: float                     # free-running / center frequency [Hz]
    gain: float                   # Kvco [Hz/V] or Kdco [Hz/LSB]
    pn_dbchz: float = -110.0      # spot phase noise on the 1/f^2 asymptote
    pn_foffset: float = 1e6       # offset of the spot [Hz]
    pn_f1f3: float = 2e5          # 1/f^3 corner [Hz]
    pn_floor_dbchz: float = -150.0
    # ---- second-order effects (all default off) ----
    nl1: float = 0.0              # Kvco nonlinearity: f = f0 + gain·v·(1+nl1·v+nl2·v²)
    nl2: float = 0.0
    pushing_hz_v: float = 0.0     # supply pushing [Hz per volt of supply ripple]
    band_step_hz: float = 0.0     # coarse-tuning band pitch (0 = single band)
    n_bands: int = 1              # number of coarse bands, centered on f0

    def leeson(self, name: str = "vco") -> LeesonOscillator:
        return LeesonOscillator.from_spot(name, self.pn_dbchz, self.pn_foffset,
                                          f_1f3=self.pn_f1f3,
                                          floor_dbchz=self.pn_floor_dbchz)

    def freq_law(self, v: float, band: int = 0) -> float:
        """f(v, band) including Kvco nonlinearity and coarse band offset."""
        f_band = self.f0 + (band - (self.n_bands - 1) / 2.0) * self.band_step_hz \
            if self.n_bands > 1 else self.f0
        return f_band + self.gain * v * (1.0 + self.nl1 * v + self.nl2 * v * v)

    def kvco_at(self, v: float) -> float:
        """Local small-signal gain df/dv at operating point v."""
        return self.gain * (1.0 + 2.0 * self.nl1 * v + 3.0 * self.nl2 * v * v)

    def v_for(self, f_target: float, band: int | None = None) -> float:
        """Control voltage solving freq_law(v, band) = f_target (real root
        nearest the linear estimate)."""
        if band is None:
            band = (self.n_bands - 1) // 2 if self.n_bands > 1 else 0
        f_band = self.f0 + (band - (self.n_bands - 1) / 2.0) * self.band_step_hz \
            if self.n_bands > 1 else self.f0
        target = (f_target - f_band) / self.gain
        if self.nl1 == 0.0 and self.nl2 == 0.0:
            return target
        roots = np.roots([self.nl2, self.nl1, 1.0, -target])
        real = roots[np.abs(roots.imag) < 1e-9].real
        if real.size == 0:
            return target
        return float(real[np.argmin(np.abs(real - target))])


class Oscillator:
    def __init__(self, cfg: OscConfig, fs: float, rng: np.random.Generator,
                 noise: bool = True, name: str = "vco"):
        self.cfg = cfg
        self.noise_on = noise
        self.gen = OscPhaseNoiseGen(cfg.leeson(name), fs, rng) if noise else None
        self.phi_acc_noise = 0.0     # accumulated (random-walk) phase noise [rad]
        self.band = (cfg.n_bands - 1) // 2 if cfg.n_bands > 1 else 0

    def freq(self, ctrl: float, v_supply: float = 0.0) -> float:
        f = self.cfg.freq_law(ctrl, self.band)
        if self.cfg.pushing_hz_v != 0.0:
            f += self.cfg.pushing_hz_v * v_supply
        return f

    def noise_step(self) -> float:
        """Total oscillator phase-noise sample for this step [rad]."""
        if not self.noise_on:
            return 0.0
        d, add = self.gen.step()
        self.phi_acc_noise += d
        return self.phi_acc_noise + add

    def noise_steps(self, n: int) -> np.ndarray:
        if not self.noise_on:
            return np.zeros(n)
        d, add = self.gen.steps(n)
        walk = self.phi_acc_noise + np.cumsum(d)
        self.phi_acc_noise = float(walk[-1])
        return walk + add
