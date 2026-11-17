# Vendored from PA_DPD@a9220d6: src/padpd/dpd/ila.py
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

Iterating is not monotone. Near saturation the first predistorter asks
for peaks the PA cannot deliver; the next post-inverse is then fitted
outside the range it was trained on and the error grows with every pass.
GMP-ILA on ReferencePA, 80 MHz 256-QAM, EVM after 1 / 2 / 4 iterations:
drive 0.14 -63.5 / -67.2 / -65.5 dB, but drive 0.18 -38.3 / -28.7 /
-22.6 and drive 0.22 -24.6 / -11.2 / +1.9 (measured 2026-11-11; issue
#4, where mmw60_trx saw the same on a soft-compressing Rapp PA). A fixed
iteration count therefore published a divergence as "this operating
point cannot be linearized" (docs/05 section 2.2). ``fit`` now measures
every predistorter on the PA and stops once an iteration is clearly worse
than the best so far, keeping the last one that was not.

"Clearly" is ``keep_best_tol_db`` (1 dB). The check is the NMSE on the
training signal, and near the floor it does not rank two good iterations
the way the validation EVM does: at 320 MHz / 4096-QAM, 10 symbols,
drive 0.14, iteration 2 read 0.7 dB worse on the check (-57.9 vs -58.6)
but 1.2 dB better in validation EVM (-64.2 vs -63.0). So a later
iteration within the tolerance is kept; one beyond it is a divergence
(at 20 symbols the same setup read 7.8 dB worse, and keeping iteration 1
moves the validation EVM from -50.4 to -58.0 dB).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..pa.base import PAModel
from ..pa.gmp import GMPModel


def _error_db(y: np.ndarray, ref: np.ndarray) -> float:
    """NMSE of the observed PA output against the linear target, dB."""
    err = np.mean(np.abs(y - ref) ** 2) / np.mean(np.abs(ref) ** 2)
    return float(10 * np.log10(max(err, 1e-300)))


class ILAPredistorter:
    def __init__(self, model_factory: Callable[[], PAModel] | None = None,
                 n_iterations: int = 2,
                 target_gain: complex | None = None,
                 fit_kwargs: dict | None = None,
                 keep_best: bool = True,
                 keep_best_tol_db: float = 1.0):
        """``target_gain`` fixes the linearization target G in y ≈ G*x.

        By default G is estimated as the least-squares scalar gain of the
        first PA pass. Pass an explicit value to use another convention
        (e.g. OpenDPD's peak-amplitude ratio, ``target_gain_opendpd``).

        ``fit_kwargs`` are forwarded to the post-inverse model's ``fit``
        (e.g. ``{"regularization": 1e-9}`` for ridge-stabilized LS on
        ill-conditioned measured data).

        ``keep_best`` (default) measures each predistorter on the PA -
        the NMSE of pa(dpd(x)) against G*x on the training signal. A
        model more than ``keep_best_tol_db`` worse than the best so far
        ends the loop and the previous kept model stays; otherwise the
        newest model is kept (the module docstring says why not the
        minimum). That costs one PA pass more than ``n_iterations``
        (the last model's own check); the other checks are the next
        iteration's pass anyway. ``n_iterations`` becomes an upper
        bound. ``False`` restores the fixed-count loop exactly: the
        last model, no extra pass. After ``fit``, ``history`` holds the
        observed error per pass (``history[0]`` is without DPD) and
        ``best_iteration`` the 1-based iteration whose model was kept.
        """
        self.model_factory = model_factory or GMPModel
        self.n_iterations = n_iterations
        self.keep_best = keep_best
        self.keep_best_tol_db = float(keep_best_tol_db)
        self.history: list[float] = []
        self.best_iteration: int | None = None
        self.dpd_model: PAModel | None = None
        self.target_gain: complex | None = target_gain
        self.fit_kwargs = fit_kwargs or {}

    def fit(self, pa: Callable[[np.ndarray], np.ndarray],
            x: np.ndarray) -> ILAPredistorter:
        """Learn the predistorter against a black-box PA on signal ``x``."""
        if not self.keep_best:
            return self._fit_fixed(pa, x)
        u = x
        self.history = []
        best = None                      # lowest check error so far, dB
        kept = None                      # (model, iteration)
        for it in range(self.n_iterations + 1):
            y = pa(u)
            if it == 0 and self.target_gain is None:
                # Target linear gain: least-squares scalar of the first pass.
                self.target_gain = complex(np.vdot(x, y) / np.vdot(x, x))
            err = _error_db(y, self.target_gain * x)
            self.history.append(err)
            if it > 0:                   # this pass measured model `it`
                if best is not None and err > best + self.keep_best_tol_db:
                    break                # diverging: later passes only worsen
                best = err if best is None else min(best, err)
                kept = (self.dpd_model, it)
            if it == self.n_iterations:
                break
            model = self.model_factory()
            model.fit(y / self.target_gain, u, **self.fit_kwargs)
            self.dpd_model = model
            u = model(x)
        self.dpd_model, self.best_iteration = kept
        return self

    def _fit_fixed(self, pa, x) -> ILAPredistorter:
        """The fixed-count loop: keep the last model, never measure it."""
        u = x
        self.history = []
        for it in range(self.n_iterations):
            y = pa(u)
            if it == 0 and self.target_gain is None:
                self.target_gain = complex(np.vdot(x, y) / np.vdot(x, x))
            self.history.append(_error_db(y, self.target_gain * x))
            model = self.model_factory()
            model.fit(y / self.target_gain, u, **self.fit_kwargs)
            self.dpd_model = model
            if it < self.n_iterations - 1:   # final pass's u is never used
                u = model(x)
        self.best_iteration = self.n_iterations
        return self

    def fit_measured(self, x: np.ndarray, y: np.ndarray) -> ILAPredistorter:
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
        """Persist the fitted predistorter (model + target gain) as .npz.

        Only linear-in-parameters models (coeffs + get_config) persist
        here; a neural predistorter saves through its own ``save()``.
        """
        if self.dpd_model is None:
            raise RuntimeError("predistorter is not fitted; nothing to save")
        if getattr(self.dpd_model, "coeffs", None) is None:
            raise TypeError(
                f"{type(self.dpd_model).__name__} is not a coefficients "
                "model; persist it with its own save() instead")
        np.savez(path,
                 class_name=type(self.dpd_model).__name__,
                 config=repr(self.dpd_model.get_config()),
                 coeffs=self.dpd_model.coeffs,
                 target_gain=np.complex128(self.target_gain),
                 n_iterations=self.n_iterations)

    @classmethod
    def load(cls, path: str) -> ILAPredistorter:
        """Load a predistorter saved with :meth:`save`.

        Restores the model factory (same class/config as the saved
        model) and iteration count, so a later ``fit()`` refits the same
        structure instead of silently falling back to the defaults.
        """
        from ..pa import load_model
        model = load_model(path)
        d = np.load(path, allow_pickle=False)
        dpd = cls(model_factory=lambda m=model: type(m)(**m.get_config()),
                  n_iterations=(int(d["n_iterations"])
                                if "n_iterations" in d else 2))
        dpd.dpd_model = model
        dpd.target_gain = complex(d["target_gain"])
        return dpd
