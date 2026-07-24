# Vendored from PA_DPD@44f9ee99: src/padpd/pa/memory_polynomial.py
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

    def basis_matrix(self, x: np.ndarray) -> np.ndarray:
        cols = []
        for m in range(self.memory_depth):
            xm = delayed(x, m)
            am = np.abs(xm)
            for k in range(self.order):
                cols.append(xm * am**k)
        return np.stack(cols, axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray,
            regularization: float = 0.0) -> "MemoryPolynomialModel":
        self.coeffs = lstsq_fit(self.basis_matrix(x), y, regularization)
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.coeffs is None:
            raise RuntimeError("model is not fitted; call fit(x, y) first")
        return self.basis_matrix(x) @ self.coeffs
