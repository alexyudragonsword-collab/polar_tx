# Vendored from PA_DPD@44f9ee99: src/padpd/pa/saleh.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Saleh memoryless AM-AM / AM-PM model.

Classic TWT-style model, useful for early sanity checks:

    A(r)   = alpha_a * r / (1 + beta_a * r^2)      (AM-AM)
    Phi(r) = alpha_p * r^2 / (1 + beta_p * r^2)    (AM-PM, radians)
"""

from __future__ import annotations

import numpy as np

from .base import PAModel


class SalehPA(PAModel):
    def __init__(self, alpha_a: float = 2.1587, beta_a: float = 1.1517,
                 alpha_p: float = 4.0033, beta_p: float = 9.1040):
        self.alpha_a = alpha_a
        self.beta_a = beta_a
        self.alpha_p = alpha_p
        self.beta_p = beta_p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        r = np.abs(x)
        amp = self.alpha_a * r / (1 + self.beta_a * r**2)
        phase = self.alpha_p * r**2 / (1 + self.beta_p * r**2)
        # Preserve input phase, add AM-PM rotation.
        unit = np.where(r > 0, x / np.where(r > 0, r, 1), 0)
        return amp * unit * np.exp(1j * phase)

    def get_config(self) -> dict:
        return {"alpha_a": self.alpha_a, "beta_a": self.beta_a,
                "alpha_p": self.alpha_p, "beta_p": self.beta_p}

    @property
    def small_signal_gain(self) -> float:
        return self.alpha_a
