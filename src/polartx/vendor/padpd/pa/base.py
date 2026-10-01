# Vendored from PA_DPD@44cbcb3: src/padpd/pa/base.py
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

    def fit(self, x: np.ndarray, y: np.ndarray) -> PAModel:
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
              regularization: float = 0.0, *,
              weights: np.ndarray | None = None,
              penalty: np.ndarray | None = None,
              penalty_weight: float = 0.0) -> np.ndarray:
    """Solve y ~= phi @ w by (optionally ridge-regularized) least squares.

    ``regularization`` is a relative Tikhonov factor: the ridge term is
    ``regularization * mean(diag(phi^H phi))``, so it is invariant to
    signal scale. Ill-conditioned polynomial bases (condition numbers of
    1e7+ on measured PA data) otherwise produce huge, delicately
    cancelling coefficients that blow up on memory-warmup boundaries.

    Keyword-only extensions (defaults reproduce the plain solver):

    - ``weights``: per-sample non-negative WLS weights — de-emphasize
      low-SNR feedback samples or corrupted segments, or emphasize the
      rare high-amplitude peaks that dominate spectral regrowth.
    - ``penalty`` (Q, n) with ``penalty_weight``: structural Tikhonov
      term ``mu * P^H P``, e.g. a second-difference matrix over spline
      control points (the P-spline roughness penalty). ``penalty_weight``
      is relative like ``regularization``: ``mu = penalty_weight *
      trace(phi^H phi) / trace(P^H P)``, so it is scale-invariant.
    """
    if weights is not None:
        sw = np.sqrt(np.asarray(weights, dtype=float).reshape(-1))
        phi = phi * sw[:, None]
        y = y * sw
    if penalty is not None and penalty_weight > 0 and penalty.size:
        a = phi.conj().T @ phi
        n = a.shape[0]
        ptp = penalty.conj().T @ penalty
        mu = penalty_weight * np.trace(a).real / max(np.trace(ptp).real,
                                                     1e-30)
        if regularization > 0:
            a = a + (regularization * np.trace(a).real / n) * np.eye(n)
        return np.linalg.solve(a + mu * ptp, phi.conj().T @ y)
    if regularization > 0:
        a = phi.conj().T @ phi
        lam = regularization * np.trace(a).real / a.shape[0]
        return np.linalg.solve(a + lam * np.eye(a.shape[0]),
                               phi.conj().T @ y)
    coeffs, *_ = np.linalg.lstsq(phi, y, rcond=None)
    return coeffs


def basis_cond(model, x: np.ndarray, n_samples: int = 8192) -> float:
    """Condition number of the model's design matrix on a signal slice.

    The practical health check before trusting an LS fit: polynomial
    envelope bases routinely reach 1e7+ where B-spline bases stay within
    a few orders of magnitude.
    """
    phi = model.basis_matrix(np.asarray(x)[:n_samples])
    return float(np.linalg.cond(phi))


def nmse_db(y_ref: np.ndarray, y_est: np.ndarray) -> float:
    """Normalized mean squared error in dB between two complex sequences."""
    err = np.abs(y_ref - y_est) ** 2
    return float(10 * np.log10(err.sum() / (np.abs(y_ref) ** 2).sum()))
