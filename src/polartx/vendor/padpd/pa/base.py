# Vendored from PA_DPD@44f9ee99: src/padpd/pa/base.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Common interface for PA behavioral models.

Every model maps a complex baseband input sequence to a complex output
sequence via ``__call__``. Trainable models additionally implement
``fit(x, y)``. Neural models added in later phases should follow the same
interface so they can be swapped against the GMP baseline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PAModel(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the model to a complex baseband sequence."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PAModel":
        raise NotImplementedError(f"{type(self).__name__} is not trainable")

    def get_config(self) -> dict:
        """Constructor kwargs needed to re-instantiate this model."""
        raise NotImplementedError(f"{type(self).__name__} has no get_config")

    def save(self, path: str) -> None:
        """Persist the model (class, config, fitted coefficients) as .npz.

        Reload with :func:`padpd.pa.load_model`. This is also the handoff
        format for coefficient download to FPGA/ASIC implementations.
        """
        payload = {
            "class_name": type(self).__name__,
            "config": repr(self.get_config()),
        }
        coeffs = getattr(self, "coeffs", None)
        if coeffs is not None:
            payload["coeffs"] = coeffs
        np.savez(path, **payload)


def lstsq_fit(phi: np.ndarray, y: np.ndarray,
              regularization: float = 0.0) -> np.ndarray:
    """Solve y ~= phi @ w by (optionally ridge-regularized) least squares.

    ``regularization`` is a relative Tikhonov factor: the ridge term is
    ``regularization * mean(diag(phi^H phi))``, so it is invariant to
    signal scale. Ill-conditioned polynomial bases (condition numbers of
    1e7+ on measured PA data) otherwise produce huge, delicately
    cancelling coefficients that blow up on memory-warmup boundaries.
    """
    if regularization > 0:
        a = phi.conj().T @ phi
        lam = regularization * np.trace(a).real / a.shape[0]
        return np.linalg.solve(a + lam * np.eye(a.shape[0]),
                               phi.conj().T @ y)
    coeffs, *_ = np.linalg.lstsq(phi, y, rcond=None)
    return coeffs


def nmse_db(y_ref: np.ndarray, y_est: np.ndarray) -> float:
    """Normalized mean squared error in dB between two complex sequences."""
    err = np.abs(y_ref - y_est) ** 2
    return float(10 * np.log10(err.sum() / (np.abs(y_ref) ** 2).sum()))
