# Vendored from PA_DPD@44cbcb3: src/padpd/pa/memory_polynomial.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Memory Polynomial (MP) model with least-squares fitting.

Basis functions: phi_{k,m}(n) = x(n-m) * |x(n-m)|^k
for k = 0..order-1 and m = 0..memory_depth-1.
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


class MemoryPolynomialModel(PAModel):
    def __init__(self, order: int = 7, memory_depth: int = 4):
        if order < 1 or memory_depth < 1:
            raise ValueError("order and memory_depth must be >= 1")
        self.order = order
        self.memory_depth = memory_depth
        self.coeffs: np.ndarray | None = None

    def get_config(self) -> dict:
        return {"order": self.order, "memory_depth": self.memory_depth}

    @property
    def n_coeffs(self) -> int:
        return self.order * self.memory_depth

    def basis_matrix(self, x: np.ndarray) -> np.ndarray:
        cols = []
        for m in range(self.memory_depth):
            xm = delayed(x, m)
            am = np.abs(xm)
            for k in range(self.order):
                cols.append(xm * am**k)
        return np.stack(cols, axis=1)

    def branch_delays(self) -> list[tuple[int, int]]:
        """(carrier delay, envelope delay) per branch — LUT metadata."""
        return [(m, m) for m in range(self.memory_depth)]

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
