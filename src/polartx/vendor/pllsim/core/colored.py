# Vendored from pll_simulator@d7be4712: src/pllsim/core/colored.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Time-domain colored-noise synthesis matched to target PSDs.

synth_from_psd() produces a real sequence whose one-sided PSD (as measured by
Welch) matches psd_func(f) [unit^2/Hz].  Used to inject reference flicker,
oscillator flicker-FM and any tabulated profile into time-domain simulations.

OscPhaseNoiseGen decomposes a Leeson profile into per-cycle contributions:
  * 1/f^2 (white FM)  -> i.i.d. Gaussian phase-increment random walk
  * 1/f^3 (flicker FM)-> pre-synthesized 1/f frequency-noise sequence, accumulated
  * floor (white PM)  -> additive, non-accumulated phase
"""
from __future__ import annotations

import numpy as np

TWOPI = 2.0 * np.pi


def synth_from_psd(psd_func, fs: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Synthesize n samples at rate fs with one-sided PSD psd_func(f) [u^2/Hz].

    FFT-domain shaping of complex white Gaussian noise; Hermitian symmetry
    enforced by irfft.  The DC bin is zeroed (no information below fs/n).
    """
    nf = n // 2 + 1
    f = np.arange(nf) * (fs / n)
    amp = np.zeros(nf)
    if nf > 1:
        s = np.asarray(psd_func(f[1:]), dtype=float)
        s = np.maximum(s, 0.0)
        # X_k drawn so that E|X_k|^2 * 2/(fs*n) = S(f_k)  (one-sided Welch scaling)
        amp[1:] = np.sqrt(s * fs * n / 2.0)
    x = amp * (rng.standard_normal(nf) + 1j * rng.standard_normal(nf)) / np.sqrt(2.0)
    x[0] = 0.0
    if n % 2 == 0:
        x[-1] = np.real(x[-1]) * np.sqrt(2.0)
    return np.fft.irfft(x, n=n)


class OscPhaseNoiseGen:
    """Per-cycle oscillator phase-noise generator for a Leeson profile.

    Called once per engine step (rate fs = step rate, usually fref).  Returns
    (d_phi_accum, phi_add):
      d_phi_accum : phase increment to ADD to the accumulated (random-walk) phase
      phi_add     : additive white PM (not accumulated)

    Profile: S_phi(f) = k2/f^2 * (1 + f_1f3/f) + floor    [rad^2/Hz DSB]

    White-FM part: S_phi = k2/f^2 corresponds to independent per-step phase
    increments with variance sigma^2 = k2 * (2*pi^2)/fs... derivation:
    a random walk phi[n] = sum(w[k]), w iid with var sw, has
    S_phi(f) = sw / (fs * |2 sin(pi f/fs)|^2) * ... -> low-f asymptote
    S_phi(f) ≈ sw * fs / (2*pi*f)^2 * (2/fs)^... : with our one-sided DSB
    convention S_phi(f) = 2*sw/(fs*(2*pi f/fs)^2) = sw*fs/(2*pi^2 f^2)
    => sw = 2*pi^2 * k2 / fs.
    """

    def __init__(self, profile, fs: float, rng: np.random.Generator,
                 chunk: int = 1 << 16):
        self.fs = fs
        self.rng = rng
        self.k2 = getattr(profile, "k2", 0.0)
        self.f_1f3 = getattr(profile, "f_1f3", 0.0)
        self.floor = getattr(profile, "floor", 0.0)
        self.sw = 2.0 * np.pi**2 * self.k2 / fs          # white-FM increment var [rad^2]
        self.s_pm = self.floor * fs / 2.0                # white-PM var [rad^2]
        self.chunk = chunk
        self._flick_buf = np.zeros(0)
        self._idx = 0

    def _refill_flicker(self):
        """1/f^3 phase region == 1/f frequency noise; synthesize df sequence."""
        if self.k2 <= 0 or self.f_1f3 <= 0:
            self._flick_buf = np.zeros(self.chunk)
        else:
            # target: S_phi_flick(f) = k2*f_1f3/f^3 = (2*pi)^2 S_df(f) / (2*pi f)^2 ...
            # phase increment per step dphi = 2*pi*df/fs; want S_dphi_walk low-f
            # asymptote to match: random walk of increments w[n] with PSD S_w(f)
            # gives S_phi ≈ S_w(f)*fs^2/(2*pi f)^2 ... simpler: synthesize w with
            # S_w(f) = k2*f_1f3*(2*pi)^2/(fs^2 * f) ... then S_phi = S_w*(fs/(2 pi f))^2
            c = self.k2 * self.f_1f3 * (TWOPI / self.fs) ** 2
            self._flick_buf = synth_from_psd(lambda f: c / f, self.fs, self.chunk, self.rng)
        self._idx = 0

    def step(self) -> tuple[float, float]:
        if self._idx >= self._flick_buf.size:
            self._refill_flicker()
        w_flick = self._flick_buf[self._idx]
        self._idx += 1
        d_phi = 0.0
        if self.sw > 0:
            d_phi += self.rng.normal(0.0, np.sqrt(self.sw))
        d_phi += w_flick
        phi_add = self.rng.normal(0.0, np.sqrt(self.s_pm)) if self.s_pm > 0 else 0.0
        return d_phi, phi_add

    def steps(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized: n per-cycle (d_phi_accum, phi_add) samples."""
        d = np.zeros(n)
        if self.sw > 0:
            d += self.rng.normal(0.0, np.sqrt(self.sw), n)
        if self.k2 > 0 and self.f_1f3 > 0:
            need = n
            pos = 0
            out = np.empty(n)
            while need > 0:
                if self._idx >= self._flick_buf.size:
                    self._refill_flicker()
                take = min(need, self._flick_buf.size - self._idx)
                out[pos:pos + take] = self._flick_buf[self._idx:self._idx + take]
                self._idx += take
                pos += take
                need -= take
            d += out
        add = self.rng.normal(0.0, np.sqrt(self.s_pm), n) if self.s_pm > 0 else np.zeros(n)
        return d, add
