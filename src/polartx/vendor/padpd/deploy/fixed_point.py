# Vendored from PA_DPD@44f9ee99: src/padpd/deploy/fixed_point.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Fixed-point (bit-true) evaluation of linear-in-parameters models.

A model like ``y = Phi @ w`` (MP / GMP / DDR-Volterra) is evaluated on
hardware as: form the basis columns Phi (from a quantized input), quantize
each column and the coefficients to a chosen bit width, then accumulate the
products in a wide accumulator. This module reproduces that datapath in
numpy so the quantization loss can be measured before committing to RTL.

Quantizer: symmetric signed, power-of-two step size (as in OpenDPD's
``round_scale2pow2`` and typical DPD ASICs) so scaling is a bit shift.
Complex values share one format across I and Q (as in real IQ datapaths).

This is a behavioral bit-true model of the arithmetic, not a full RTL
timing/resource model; the accumulator is treated as full precision
(wide enough not to overflow), which matches well-designed DPD hardware.
"""

from __future__ import annotations

import numpy as np


def _pow2_step(peak: float, n_bits: int) -> float:
    """Power-of-two quantization step for a symmetric signed n_bits format
    covering ``peak``."""
    qmax = 2 ** (n_bits - 1) - 1
    if peak <= 0:
        return 1.0
    return 2.0 ** np.ceil(np.log2(peak / qmax))


def quantize_symmetric(x: np.ndarray, n_bits: int) -> np.ndarray:
    """Quantize real or complex ``x`` to a symmetric signed n_bits format.

    A single power-of-two step is chosen from the array peak; complex
    inputs share the step across real and imaginary parts.
    """
    x = np.asarray(x)
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -(2 ** (n_bits - 1))
    if np.iscomplexobj(x):
        peak = max(np.abs(x.real).max(), np.abs(x.imag).max())
        step = _pow2_step(peak, n_bits)
        r = np.clip(np.round(x.real / step), qmin, qmax) * step
        i = np.clip(np.round(x.imag / step), qmin, qmax) * step
        return r + 1j * i
    peak = np.abs(x).max()
    step = _pow2_step(peak, n_bits)
    return np.clip(np.round(x / step), qmin, qmax) * step


class FixedPointPolyModel:
    """Bit-true wrapper around a fitted linear-params model.

    ``model`` must expose ``basis_matrix(x)`` and a fitted ``coeffs``
    (MP / GMP / DDR-Volterra all do). Coefficients are quantized once at
    construction; the input, each basis column, and the output are
    quantized per call. Basis columns are quantized independently (each
    tap has its own datapath format), which is what real hardware does
    and is essential because ``|x|^0`` and ``|x|^k`` differ by orders of
    magnitude.
    """

    def __init__(self, model, w_bits: int = 16, sig_bits: int = 16):
        if getattr(model, "coeffs", None) is None:
            raise ValueError("model must be fitted before quantization")
        self.model = model
        self.w_bits = w_bits
        self.sig_bits = sig_bits
        self.wq = quantize_symmetric(model.coeffs, w_bits)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        xq = quantize_symmetric(np.asarray(x, dtype=complex), self.sig_bits)
        phi = self.model.basis_matrix(xq)
        # per-column (per-tap) activation quantization
        phiq = np.empty_like(phi)
        for c in range(phi.shape[1]):
            phiq[:, c] = quantize_symmetric(phi[:, c], self.sig_bits)
        return phiq @ self.wq  # full-precision accumulate

    @property
    def n_coeffs(self) -> int:
        return len(self.wq)


def mac_cost(n_coeffs: int, sample_rate_hz: float) -> dict:
    """Hardware cost estimate for a linear-params model.

    One complex coefficient = one complex MAC/sample = 4 real multiplies
    (3 with a Karatsuba-style complex multiplier). Returns real-MAC rate
    at the given sample rate.
    """
    real_macs_per_sample = 4 * n_coeffs
    return {
        "complex_macs_per_sample": n_coeffs,
        "real_macs_per_sample": real_macs_per_sample,
        "real_gmac_per_s": real_macs_per_sample * sample_rate_hz / 1e9,
    }
