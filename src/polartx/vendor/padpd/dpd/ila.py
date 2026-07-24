# Vendored from PA_DPD@44f9ee99: src/padpd/dpd/ila.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Indirect Learning Architecture (ILA) digital predistortion.

Classic industrial flow:

    1. Drive the PA with u(n) (initially u = x).
    2. Fit a post-inverse model:  y(n)/G  ->  u(n)   (G = target linear gain)
    3. Copy the post-inverse in front of the PA as the predistorter and
       iterate with u = DPD(x).

The predistorter can be any trainable :class:`~padpd.pa.PAModel`
(GMP by default). The ``pa`` argument is a black box callable — the
ReferencePA today, a Cadence/measurement replay tomorrow.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..pa.base import PAModel
from ..pa.gmp import GMPModel


class ILAPredistorter:
    def __init__(self, model_factory: Callable[[], PAModel] | None = None,
                 n_iterations: int = 2,
                 target_gain: complex | None = None,
                 fit_kwargs: dict | None = None):
        """``target_gain`` fixes the linearization target G in y ≈ G*x.

        By default G is estimated as the least-squares scalar gain of the
        first PA pass. Pass an explicit value to use another convention
        (e.g. OpenDPD's peak-amplitude ratio, ``target_gain_opendpd``).

        ``fit_kwargs`` are forwarded to the post-inverse model's ``fit``
        (e.g. ``{"regularization": 1e-9}`` for ridge-stabilized LS on
        ill-conditioned measured data).
        """
        self.model_factory = model_factory or GMPModel
        self.n_iterations = n_iterations
        self.dpd_model: PAModel | None = None
        self.target_gain: complex | None = target_gain
        self.fit_kwargs = fit_kwargs or {}

    def fit(self, pa: Callable[[np.ndarray], np.ndarray],
            x: np.ndarray) -> "ILAPredistorter":
        """Learn the predistorter against a black-box PA on signal ``x``."""
        u = x
        for it in range(self.n_iterations):
            y = pa(u)
            if it == 0 and self.target_gain is None:
                # Target linear gain: least-squares scalar of the first pass.
                self.target_gain = complex(np.vdot(x, y) / np.vdot(x, x))
            model = self.model_factory()
            model.fit(y / self.target_gain, u, **self.fit_kwargs)
            self.dpd_model = model
            u = model(x)
        return self

    def fit_measured(self, x: np.ndarray, y: np.ndarray) -> "ILAPredistorter":
        """Single-shot ILA from a measured input/output pair.

        Fits the post-inverse directly on measured data (y/G -> x) without
        running any PA model in the loop — the protocol used by OpenDPD's
        classical benchmark. Use this when you have a captured dataset but
        no executable PA; use :meth:`fit` when a PA (model or testbench)
        can be driven iteratively.
        """
        if self.target_gain is None:
            self.target_gain = complex(np.vdot(x, y) / np.vdot(x, x))
        model = self.model_factory()
        model.fit(y / self.target_gain, x, **self.fit_kwargs)
        self.dpd_model = model
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Predistort a signal."""
        if self.dpd_model is None:
            raise RuntimeError("predistorter is not fitted; call fit() first")
        return self.dpd_model(x)

    def linearize(self, pa: Callable[[np.ndarray], np.ndarray],
                  x: np.ndarray) -> np.ndarray:
        """Convenience: PA output with DPD applied, i.e. pa(dpd(x))."""
        return pa(self(x))

    def save(self, path: str) -> None:
        """Persist the fitted predistorter (model + target gain) as .npz."""
        if self.dpd_model is None:
            raise RuntimeError("predistorter is not fitted; nothing to save")
        np.savez(path,
                 class_name=type(self.dpd_model).__name__,
                 config=repr(self.dpd_model.get_config()),
                 coeffs=self.dpd_model.coeffs,
                 target_gain=np.complex128(self.target_gain))

    @classmethod
    def load(cls, path: str) -> "ILAPredistorter":
        """Load a predistorter saved with :meth:`save`."""
        from ..pa import load_model
        dpd = cls()
        dpd.dpd_model = load_model(path)
        d = np.load(path, allow_pickle=False)
        dpd.target_gain = complex(d["target_gain"])
        return dpd
