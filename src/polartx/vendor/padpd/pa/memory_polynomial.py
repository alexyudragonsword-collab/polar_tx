# Vendored from PA_DPD@a9220d6: src/padpd/pa/memory_polynomial.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Memory Polynomial (MP) model with least-squares fitting.

Basis functions: phi_{k,m}(n) = x(n-m*s) * |x(n-m*s)|^k
for k = 0..order-1, m = 0..memory_depth-1 and tap spacing s.

``tap_spacing`` (default 1) puts the memory taps s samples apart, so the
memory span stays fixed in seconds when the sampling rate rises. A PA
whose memory is defined in time (``ReferencePA(fs=..., fir_fs=...)``)
needs it. Measured on an 80 MHz signal at 320 MS/s against a PA whose
taps sit 12.5 ns (4 samples) apart (issue #4): MP order 7, memory 4
fits to -27.3 dB with sample-spaced taps and -36.4 dB with
tap_spacing 4; a GMP-ILA DPD reaches an EVM of -28.6 dB and -46.4 dB.

That number is for a DPD driving the PA directly. In a full TX chain a
baseband lowpass between the DPD and the PA adds per-sample memory, and
uniform spacing stretches every delay, so it matches one memory scale.
On mmw60_trx's chain (11be 80 MHz x K, 2x oversampling, best ILA
iteration kept) the cost of time-defined memory, counted as the extra
output backoff needed for the same EVM as a per-sample-memory PA, is
0.5 / 0.6 dB with PA taps 64 samples apart and 1.6 / 2.3 dB with them 16
apart (EVM targets -25 / -27 dB). Matched spacing is needed: with taps
64 apart, spacing 1 reaches only -25.5 dB even at 12 dB backoff, matched
spacing -27 dB at 8.9 dB. Which spacing wins depends on the backoff
(issue #4 follow-ups). Earlier readings there, taken at a single backoff
that sat on the DPD's cliff and partly with a diverged ILA iteration,
overstated the cost as 4-6 dB. A mixed delay set - a few per-sample taps
plus a few far ones - might recover part of the 16-sample case; it is
not implemented.
"""

from __future__ import annotations

import numpy as np

from .base import PAModel, lstsq_fit


def delayed(x: np.ndarray, m: int) -> np.ndarray:
    """Shift x by m samples (positive m = delay/lag), zero-padded."""
    if m == 0:
        return x
    out = np.zeros_like(x)
    if m > 0:
        out[m:] = x[:-m]
    else:
        out[:m] = x[-m:]
    return out


def check_tap_spacing(tap_spacing) -> int:
    if int(tap_spacing) != tap_spacing or tap_spacing < 1:
        raise ValueError("tap_spacing must be an integer >= 1")
    return int(tap_spacing)


class MemoryPolynomialModel(PAModel):
    def __init__(self, order: int = 7, memory_depth: int = 4,
                 tap_spacing: int = 1):
        if order < 1 or memory_depth < 1:
            raise ValueError("order and memory_depth must be >= 1")
        self.order = order
        self.memory_depth = memory_depth
        self.tap_spacing = check_tap_spacing(tap_spacing)
        self.coeffs: np.ndarray | None = None

    def get_config(self) -> dict:
        cfg = {"order": self.order, "memory_depth": self.memory_depth}
        # only when it is not the default, so a sample-spaced model saves
        # exactly as before and older readers can still load it
        if self.tap_spacing != 1:
            cfg["tap_spacing"] = self.tap_spacing
        return cfg

    @property
    def n_coeffs(self) -> int:
        return self.order * self.memory_depth

    def basis_matrix(self, x: np.ndarray) -> np.ndarray:
        cols = []
        for m in range(self.memory_depth):
            xm = delayed(x, m * self.tap_spacing)
            am = np.abs(xm)
            for k in range(self.order):
                cols.append(xm * am**k)
        return np.stack(cols, axis=1)

    def branch_delays(self) -> list[tuple[int, int]]:
        """(carrier delay, envelope delay) per branch — LUT metadata."""
        return [(m * self.tap_spacing, m * self.tap_spacing)
                for m in range(self.memory_depth)]

    def gain_curve(self, r: np.ndarray) -> np.ndarray:
        """Complex gain of each memory tap vs envelope: sum_k c_km r^k."""
        if self.coeffs is None:
            raise RuntimeError("model is not fitted; call fit(x, y) first")
        r = np.asarray(r, dtype=float)
        powers = np.stack([r**k for k in range(self.order)], axis=1)
        return np.stack([powers @ self.coeffs[m * self.order:
                                              (m + 1) * self.order]
                         for m in range(self.memory_depth)])

    def fit(self, x: np.ndarray, y: np.ndarray,
            regularization: float = 0.0) -> MemoryPolynomialModel:
        self.coeffs = lstsq_fit(self.basis_matrix(x), y, regularization)
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.coeffs is None:
            raise RuntimeError("model is not fitted; call fit(x, y) first")
        return self.basis_matrix(x) @ self.coeffs
