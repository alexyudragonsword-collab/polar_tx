# Vendored from pll_simulator@931cfaf: src/pllsim/blocks/oscillator.py
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
from ..core.jit import kernel
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
    # ---- control-voltage range ----
    # None = unlimited, which is what every existing configuration gets and what
    # the loop assumed implicitly.  An unlimited varactor is why a coarse band
    # bank could be configured and still have nothing to do: one band reached any
    # frequency, so a band search always succeeded on the band it started in.
    # A real varactor is limited by the supply, and that limit is the whole reason
    # the capacitor bank exists.
    v_min: float | None = None
    v_max: float | None = None
    # ---- injection pulling (Adler) ----
    # An aggressor at f0 +/- pull_offset_hz drags the tank.  Outside the lock
    # range Adler's equation dphi/dt = dw - wL sin(phi) leaves, for weak
    # pulling, a residual FM of peak deviation pull_lock_range_hz at the beat
    # frequency -- so the sideband is f_L/(2*df), before the loop gets to
    # suppress whatever falls inside its bandwidth.  This is the coupling that
    # puts a PA harmonic or a neighbouring synthesizer into the spectrum
    # without touching the loop at all, which is why chasing it in the loop
    # filter never works.
    pull_lock_range_hz: float = 0.0   # f_L = f0*(Iinj/Iosc)/(2Q)
    pull_offset_hz: float = 0.0       # |f0 - f_aggressor|

    def __post_init__(self):
        # Pure validation: guiutil._revalidate re-runs this after every form
        # edit, and until it existed the fifteen fields here -- the widest
        # part of every form -- reached the engine unchecked.
        if not self.f0 > 0:
            raise ValueError(f"OscConfig f0 must be positive, got {self.f0}")
        if not self.gain or not np.isfinite(self.gain):
            raise ValueError(f"OscConfig gain must be a finite non-zero Hz/V "
                             f"(or Hz/LSB), got {self.gain}")
        if not self.pn_foffset > 0:
            raise ValueError(f"OscConfig pn_foffset must be positive, got {self.pn_foffset}")
        if self.pn_f1f3 < 0:
            raise ValueError(f"OscConfig pn_f1f3 cannot be negative, got {self.pn_f1f3}")
        if int(self.n_bands) < 1:
            raise ValueError(f"OscConfig n_bands must be >= 1, got {self.n_bands}")
        if self.n_bands > 1 and not self.band_step_hz > 0:
            raise ValueError("OscConfig band_step_hz must be positive when "
                             "n_bands > 1: a bank of coincident bands is one band")
        if self.band_step_hz < 0:
            raise ValueError(f"OscConfig band_step_hz cannot be negative, got {self.band_step_hz}")
        if self.v_min is not None and self.v_max is not None and not self.v_min < self.v_max:
            raise ValueError(f"OscConfig v_min ({self.v_min}) must be below "
                             f"v_max ({self.v_max})")
        if self.pull_lock_range_hz < 0 or self.pull_offset_hz < 0:
            raise ValueError("OscConfig pull_lock_range_hz / pull_offset_hz "
                             "cannot be negative")

    @staticmethod
    def lock_range_from_tank(f0: float, i_ratio: float, q_tank: float) -> float:
        """Adler lock range f0*(Iinj/Iosc)/(2Q), the usual way to get f_L."""
        return f0 * i_ratio / (2.0 * q_tank)

    @property
    def pulled(self) -> bool:
        return self.pull_lock_range_hz > 0.0 and self.pull_offset_hz > 0.0

    def pull_spur_dbc(self, err_at_offset: float = 1.0) -> float:
        """Pulling sideband [dBc], optionally after the loop's rejection.

        beta = f_L/df is the FM index of the residual modulation, and the
        narrowband sideband is beta/2.  Pass |err(df)| to include the loop's
        suppression of what lands in band.
        """
        if not self.pulled:
            return float("-inf")
        beta = (self.pull_lock_range_hz / self.pull_offset_hz) * err_at_offset
        return float(20.0 * np.log10(beta / 2.0))

    def within_lock_range(self) -> bool:
        """True when the aggressor would capture the oscillator outright.

        Inside the lock range there is no spur to report: the oscillator stops
        being the loop's and follows the aggressor, and the weak-pulling
        expression above does not apply at all.
        """
        return self.pulled and self.pull_offset_hz <= self.pull_lock_range_hz

    def leeson(self, name: str = "vco") -> LeesonOscillator:
        return LeesonOscillator.from_spot(name, self.pn_dbchz, self.pn_foffset,
                                          f_1f3=self.pn_f1f3,
                                          floor_dbchz=self.pn_floor_dbchz)

    def clamp_v(self, v: float) -> float:
        """Control voltage as the varactor actually sees it."""
        if self.v_min is not None:
            v = max(v, self.v_min)
        if self.v_max is not None:
            v = min(v, self.v_max)
        return v

    @property
    def v_limited(self) -> bool:
        return self.v_min is not None or self.v_max is not None

    def band_center_hz(self, band: int) -> float:
        """Free-running frequency of a coarse band."""
        if self.n_bands <= 1:
            return self.f0
        return self.f0 + (band - (self.n_bands - 1) / 2.0) * self.band_step_hz

    def band_range_hz(self, band: int) -> tuple[float, float]:
        """Frequencies a band can reach, given the control-voltage range.

        Unbounded when no range is set, which is the honest answer: with an
        unlimited varactor every band reaches every frequency.
        """
        lo_v, hi_v = self.v_min, self.v_max
        if lo_v is None or hi_v is None:      # same test as v_limited, but one
            return (-np.inf, np.inf)          # a reader and a checker can follow
        lo = self.freq_law(lo_v, band)
        hi = self.freq_law(hi_v, band)
        return (min(lo, hi), max(lo, hi))

    def band_covers(self, f_target: float, band: int) -> bool:
        """Whether a band can be tuned to a frequency without railing."""
        lo, hi = self.band_range_hz(band)
        return bool(lo <= f_target <= hi)

    def band_overlap_hz(self) -> float:
        """How far adjacent bands overlap [Hz].  Negative means a **gap**.

        The sizing rule for a coarse bank, and it is easy to get wrong in a way
        that shows up only as a frequency the loop cannot reach: a band spans
        roughly ``2 * gain * v_max`` and the bank steps by ``band_step_hz``, so
        the bank is only continuous when the span exceeds the step.  A bank with
        150 MHz of pitch and 60 MHz of span leaves 90 MHz holes, and a target in
        one of them rails whichever band the search picks -- which reads as a
        broken loop rather than as a mis-sized oscillator.
        """
        if self.n_bands <= 1 or not self.v_limited:
            return float("inf")
        lo, hi = self.band_range_hz(0)
        return float((hi - lo) - self.band_step_hz)

    def band_bank_is_continuous(self) -> bool:
        """Whether every frequency in the bank's range is reachable."""
        return self.band_overlap_hz() > 0.0

    def bank_range_hz(self) -> tuple[float, float]:
        """Frequencies the whole bank can reach, lowest band to highest."""
        if self.n_bands <= 1:
            return self.band_range_hz(0)
        lo = self.band_range_hz(0)[0]
        hi = self.band_range_hz(self.n_bands - 1)[1]
        return (lo, hi)

    def freq_law(self, v: float, band: int = 0) -> float:
        """f(v, band) including Kvco nonlinearity and coarse band offset.

        The control voltage is clamped to ``[v_min, v_max]`` when a range is set,
        so a band that cannot reach the target rails instead of pretending to.
        """
        return float(osc_freq_law(float(v), int(band), *self.law_params()))

    # ---- kernel view: the law's scalars as a compiled loop takes them
    def law_params(self) -> tuple[float, float, float, float, float, int, float, float]:
        """(f0, gain, nl1, nl2, band_step_hz, n_bands, v_lo, v_hi); an unset
        range bound is +/-inf, which clamps nothing."""
        return (float(self.f0), float(self.gain), float(self.nl1), float(self.nl2),
                float(self.band_step_hz), int(self.n_bands),
                -np.inf if self.v_min is None else float(self.v_min),
                np.inf if self.v_max is None else float(self.v_max))

    def kvco_at(self, v: float) -> float:
        """Local small-signal gain df/dv at operating point v."""
        return self.gain * (1.0 + 2.0 * self.nl1 * v + 3.0 * self.nl2 * v * v)

    def v_for(self, f_target: float, band: int | None = None) -> float:
        """Control voltage solving freq_law(v, band) = f_target (real root
        nearest the linear estimate).

        Not clamped: this is the voltage the target *would* need, so a caller can
        see that it is out of range rather than being handed a railed one.
        """
        if band is None:
            band = (self.n_bands - 1) // 2 if self.n_bands > 1 else 0
        f_band = self.band_center_hz(band)
        target = (f_target - f_band) / self.gain
        if self.nl1 == 0.0 and self.nl2 == 0.0:
            return target
        roots = np.roots([self.nl2, self.nl1, 1.0, -target])
        real = roots[np.abs(roots.imag) < 1e-9].real
        if real.size == 0:
            return target
        return float(real[np.argmin(np.abs(real - target))])


@kernel
def osc_freq_law(v: float, band: int, f0: float, gain: float, nl1: float, nl2: float,
                 band_step: float, n_bands: int, v_lo: float, v_hi: float) -> float:
    """f(v, band): clamp to the range, band offset, Kvco with nonlinearity."""
    v = max(v, v_lo)
    v = min(v, v_hi)
    f_band = f0 if n_bands <= 1 else f0 + (band - (n_bands - 1) / 2.0) * band_step
    return f_band + gain * v * (1.0 + nl1 * v + nl2 * v * v)


@kernel
def osc_freq(v: float, band: int, v_supply: float, f_pull: float, pushing_hz_v: float,
             f0: float, gain: float, nl1: float, nl2: float, band_step: float,
             n_bands: int, v_lo: float, v_hi: float) -> float:
    """The law plus supply pushing and injection pulling for one cycle."""
    f = osc_freq_law(v, band, f0, gain, nl1, nl2, band_step, n_bands, v_lo, v_hi)
    if pushing_hz_v != 0.0:
        f += pushing_hz_v * v_supply
    return f + f_pull


class Oscillator:
    def __init__(self, cfg: OscConfig, fs: float, rng: np.random.Generator,
                 noise: bool = True, name: str = "vco"):
        self.cfg = cfg
        self.noise_on = noise
        self.gen = OscPhaseNoiseGen(cfg.leeson(name), fs, rng) if noise else None
        self.phi_acc_noise = 0.0     # accumulated (random-walk) phase noise [rad]
        self.band = (cfg.n_bands - 1) // 2 if cfg.n_bands > 1 else 0

    def freq(self, ctrl: float, v_supply: float = 0.0,
             f_pull: float = 0.0) -> float:
        return float(osc_freq(float(ctrl), int(self.band), float(v_supply),
                              float(f_pull), float(self.cfg.pushing_hz_v),
                              *self.cfg.law_params()))

    def pull_hz(self, n_cycles: int, tref: float) -> np.ndarray:
        """Per-cycle frequency perturbation from an aggressor [Hz].

        Weak-pulling limit of Adler's equation: the phase error rides a
        residual modulation at the beat rate whose peak frequency deviation is
        the lock range.  Returned as a sequence so an engine can add it to the
        oscillator alongside supply pushing.
        """
        c = self.cfg
        if not c.pulled:
            return np.zeros(n_cycles)
        t = np.arange(n_cycles) * tref
        return c.pull_lock_range_hz * np.sin(2.0 * np.pi * c.pull_offset_hz * t)

    def noise_step(self) -> float:
        """Total oscillator phase-noise sample for this step [rad]."""
        if not self.noise_on or self.gen is None:
            return 0.0
        d, add = self.gen.step()
        self.phi_acc_noise += d
        return self.phi_acc_noise + add

    def noise_steps(self, n: int) -> np.ndarray:
        if not self.noise_on or self.gen is None:
            return np.zeros(n)
        d, add = self.gen.steps(n)
        walk = self.phi_acc_noise + np.cumsum(d)
        self.phi_acc_noise = float(walk[-1])
        return walk + add
