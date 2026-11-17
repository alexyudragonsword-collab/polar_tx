# Vendored from PA_DPD@a9220d6: src/padpd/pa/reference_pa.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Virtual "device under test" PA for synthetic data generation.

A Wiener-Hammerstein structure (FIR -> Saleh nonlinearity -> FIR) with
fixed coefficients. It plays the role of the transistor-level PA
(Cadence Spectre envelope simulation) until real EDA data is plugged in:
downstream modeling/DPD code treats it as a black box x -> y.

The structure is intentionally NOT in the GMP model class, so fitted GMP
models approximate it (realistic residual error) rather than match exactly.

Input is assumed to be normalized to unit average power; ``drive`` sets
the operating point (larger = closer to saturation). Small-signal gain is
normalized to ~1 so absolute gain drops out of modeling and metrics.

Note on invertibility: DPD can only linearize up to the PA's maximum
output. With unit-power OFDM (PAPR ~10-11 dB) the peak envelope is
~3.3-3.8, and the Saleh AM-AM turns over at r = 1/sqrt(beta_a). Keep
``drive`` <= ~0.15 so that peaks stay in the invertible region; around
0.22 and above the ILA DPD diverges at the peaks (a realistic failure
mode — real systems add crest factor reduction or more back-off).

This class is a *virtual test article*, deliberately NOT persistable via
``PAModel.save`` (no ``get_config``): it stands in for a transistor-level
simulation, not for a fitted model worth serializing. The same applies
to :class:`~padpd.pa.drift.DriftingReferencePA`.

Memory in time, not in samples. The FIR taps are one sample apart *at
the rate they were written for*. Called at another rate, a sample-spaced
FIR is a different PA: its memory span scales with 1/fs, so running the
same device on a signal sampled K times faster shortens its memory K
times. Give ``fs`` (the rate the PA is called at) and ``fir_fs`` (the
rate the taps are defined on) and the taps are placed ``fs / fir_fs``
samples apart, i.e. a fixed number of seconds. Which ``fir_fs`` is right
is a modelling choice: a fixed value models one device driven by wider
or narrower signals; ``fir_fs`` = 4 x the signal bandwidth reproduces
how this repository has always used the PA (the waveform generator
samples at 4 x bandwidth), i.e. memory that scales with the signal.

Measured on 2026-11-11 (memory contribution = output difference to the
same PA with each FIR collapsed to its DC gain): -20.1 dB at 20 MHz /
80 MS/s; a 20 MHz x K signal with taps fixed at 12.5 ns reads -14.7 /
-11.0 / -10.8 dB at K = 2 / 4 / 8, where sample-spaced taps stay at
-20 dB (issue #4). Only an integer ``fs / fir_fs`` is supported: a
fractional ratio needs a fractional-delay filter, whose own response
would become part of the PA.
"""

from __future__ import annotations

import numpy as np

from .base import PAModel
from .saleh import SalehPA


def tap_stride(fs: float | None, fir_fs: float | None) -> int:
    """Samples between FIR taps when a PA defined at ``fir_fs`` runs at
    ``fs``. Both None means 1 (sample-spaced, the historical behaviour)."""
    if fs is None and fir_fs is None:
        return 1
    if fs is None or fir_fs is None:
        raise ValueError("give both fs and fir_fs (the rate the PA runs at "
                         "and the rate its FIR taps are defined on), or "
                         "neither")
    if fs <= 0 or fir_fs <= 0:
        raise ValueError("fs and fir_fs must be > 0")
    ratio = fs / fir_fs
    stride = round(ratio)
    if stride < 1 or abs(ratio - stride) > 1e-9 * ratio:
        raise ValueError(
            f"fs / fir_fs = {ratio:g} is not a positive integer. A "
            "fractional ratio needs a fractional-delay filter, which would "
            "add its own response to the PA; raise the oversampling so "
            "fs is an integer multiple of fir_fs")
    return int(stride)


def spread_taps(h: np.ndarray, stride: int) -> np.ndarray:
    """Place the taps of ``h`` ``stride`` samples apart (zeros between)."""
    h = np.asarray(h)
    if stride == 1:
        return h
    out = np.zeros((len(h) - 1) * stride + 1, dtype=complex)
    out[::stride] = h
    return out


class ReferencePA(PAModel):
    def __init__(self, drive: float = 0.14,
                 fir_in: np.ndarray | None = None,
                 fir_out: np.ndarray | None = None,
                 saleh: SalehPA | None = None,
                 fs: float | None = None,
                 fir_fs: float | None = None):
        self.drive = drive
        self.fs = fs
        self.fir_fs = fir_fs
        self.tap_stride = tap_stride(fs, fir_fs)
        fir_in = (np.array([1.0, 0.12 - 0.05j, 0.04 + 0.03j])
                  if fir_in is None else np.asarray(fir_in))
        fir_out = (np.array([1.0, 0.06 + 0.025j])
                   if fir_out is None else np.asarray(fir_out))
        # the taps as applied at fs; identical arrays when stride is 1
        self.fir_in = spread_taps(fir_in, self.tap_stride)
        self.fir_out = spread_taps(fir_out, self.tap_stride)
        self.saleh = saleh or SalehPA()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        u = np.convolve(x, self.fir_in)[: len(x)]
        v = self.saleh(self.drive * u)
        w = np.convolve(v, self.fir_out)[: len(x)]
        return w / (self.drive * self.saleh.small_signal_gain)
