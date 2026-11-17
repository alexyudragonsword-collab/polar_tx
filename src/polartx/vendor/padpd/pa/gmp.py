# Vendored from PA_DPD@a9220d6: src/padpd/pa/gmp.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Generalized Memory Polynomial (GMP) model.

Extends the memory polynomial with lagging and leading envelope cross
terms (Morgan et al., 2006). This is the industrial DPD gold baseline:
all neural models should be compared against it.

Basis:
  aligned : x(n-m) |x(n-m)|^k        k=0..Ka-1, m=0..Ma-1
  lagging : x(n-m) |x(n-m-l)|^k      k=1..Kb,   m=0..Mb-1, l=1..Lb
  leading : x(n-m) |x(n-m+l)|^k      k=1..Kc,   m=0..Mc-1, l=1..Lc

``tap_spacing`` s multiplies every delay (m and l above), keeping the
memory span fixed in seconds at a higher sampling rate; see
:mod:`padpd.pa.memory_polynomial`, including why one uniform spacing does
not cover a TX lowpass and a time-defined PA at once.
"""

from __future__ import annotations

import numpy as np

from .base import PAModel, lstsq_fit
from .memory_polynomial import check_tap_spacing, delayed


class GMPModel(PAModel):
    def __init__(self, order: int = 7, memory_depth: int = 4,
                 lag_order: int = 3, lag_memory: int = 2, lag_span: int = 2,
                 lead_order: int = 3, lead_memory: int = 2,
                 lead_span: int = 2, tap_spacing: int = 1):
        if order < 1 or memory_depth < 1:
            raise ValueError("order and memory_depth must be >= 1")
        if min(lag_order, lag_memory, lag_span,
               lead_order, lead_memory, lead_span) < 0:
            raise ValueError("lag/lead orders, memories and spans "
                             "must be >= 0")
        self.order = order
        self.memory_depth = memory_depth
        self.lag_order = lag_order
        self.lag_memory = lag_memory
        self.lag_span = lag_span
        self.lead_order = lead_order
        self.lead_memory = lead_memory
        self.lead_span = lead_span
        self.tap_spacing = check_tap_spacing(tap_spacing)
        self.coeffs: np.ndarray | None = None

    def get_config(self) -> dict:
        cfg = {"order": self.order, "memory_depth": self.memory_depth,
               "lag_order": self.lag_order, "lag_memory": self.lag_memory,
               "lag_span": self.lag_span, "lead_order": self.lead_order,
               "lead_memory": self.lead_memory,
               "lead_span": self.lead_span}
        # default omitted: saved files and older readers stay unchanged
        if self.tap_spacing != 1:
            cfg["tap_spacing"] = self.tap_spacing
        return cfg

    def basis_matrix(self, x: np.ndarray) -> np.ndarray:
        a = np.abs(x)
        s = self.tap_spacing
        cols = []
        for m in range(self.memory_depth):
            xm = delayed(x, m * s)
            am = delayed(a, m * s)
            for k in range(self.order):
                cols.append(xm * am**k)
        for m in range(self.lag_memory):
            xm = delayed(x, m * s)
            for l in range(1, self.lag_span + 1):  # noqa: E741 - basis index
                al = delayed(a, (m + l) * s)
                for k in range(1, self.lag_order + 1):
                    cols.append(xm * al**k)
        for m in range(self.lead_memory):
            xm = delayed(x, m * s)
            for l in range(1, self.lead_span + 1):  # noqa: E741 - basis index
                al = delayed(a, (m - l) * s)
                for k in range(1, self.lead_order + 1):
                    cols.append(xm * al**k)
        return np.stack(cols, axis=1)

    @property
    def n_coeffs(self) -> int:
        return (self.order * self.memory_depth
                + self.lag_order * self.lag_memory * self.lag_span
                + self.lead_order * self.lead_memory * self.lead_span)

    def fit(self, x: np.ndarray, y: np.ndarray,
            regularization: float = 0.0) -> GMPModel:
        self.coeffs = lstsq_fit(self.basis_matrix(x), y, regularization)
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.coeffs is None:
            raise RuntimeError("model is not fitted; call fit(x, y) first")
        return self.basis_matrix(x) @ self.coeffs
