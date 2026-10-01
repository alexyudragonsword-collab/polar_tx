# Vendored from PA_DPD@44cbcb3: src/padpd/pa/spline_state.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""State-conditioned spline PA/DPD models (long-term memory).

Instantaneous-amplitude bases (MP/GMP/SMP) capture nanosecond electrical
memory but are blind to the *slow* mechanisms — thermal transients, bias
drift, trapping — where the effective gain depends on recent envelope
power, not the current sample. This module adds the two standard
remedies, both linear in their coefficients:

1. :class:`StateConditionedSpline` — augments the spline memory
   polynomial with slow envelope-power states

       q_k[n] = alpha_k q_k[n-1] + (1 - alpha_k) |x[n]|^2

   (logarithmically spaced time constants), spline-expanded and used
   both additively (``x[n-m] * C_i(q_k[n])``: state-dependent gain) and
   as a tap-0 amplitude-state tensor surface (``x[n] * B_j(r) *
   C_i(q_k)``: compression that *changes shape* with heating). This is
   the "additive + selected interactions" structure — a full tensor over
   every dimension would explode; the 2-D tap-0 surface keeps 16 active
   products for cubic-x-quadratic.

2. :class:`CoefficientScheduler` — for *per-capture* operating scalars
   (chamber temperature, bias setting, average power): fit one model per
   condition, then interpolate the coefficient vectors across the
   condition axis (PCHIP by default — shape-preserving, no overshoot
   between calibration points). Inside a capture the states of (1) do
   the work; across operating points the scheduler does.

Validated against :class:`~padpd.pa.thermal.ThermalReferencePA`, the
self-heating virtual DUT.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import PAModel, lstsq_fit
from .memory_polynomial import delayed
from .spline import (
    _second_difference,
    _validate_knots,
    bspline_design_matrix,
    place_knots,
)


class StateConditionedSpline(PAModel):
    """SMP + slow envelope-power state conditioning; linear in coeffs.

    ``state_alphas`` are per-sample IIR smoothing factors (one slow state
    per entry); ``q_scale`` normalizes the states into the unit interval
    where the ``state_knots`` live (calibrate it via :meth:`from_signal`).
    States are recomputed causally from block start on every call — match
    the existing warm-up convention (prepend ~200+ samples) when
    evaluating, and prefer blocks long enough for the slowest state.
    """

    def __init__(self, knots: Sequence[float] | None = None,
                 degree: int = 3, memory_depth: int = 4,
                 state_alphas: Sequence[float] = (0.99, 0.9999),
                 state_knots: Sequence[float] | None = None,
                 state_degree: int = 2, q_scale: float = 1.0,
                 interaction: bool = True,
                 n_knots: int | None = None, r_max: float = 1.0):
        if degree not in (1, 2, 3) or state_degree not in (1, 2, 3):
            raise ValueError("degree and state_degree must be 1, 2 or 3")
        if memory_depth < 1:
            raise ValueError("memory_depth must be >= 1")
        if len(state_alphas) < 1:
            raise ValueError("need at least one state alpha")
        if any(not 0.0 < a < 1.0 for a in state_alphas):
            raise ValueError("state_alphas must lie in (0, 1)")
        if q_scale <= 0:
            raise ValueError("q_scale must be > 0")
        if knots is None:
            n = 8 if n_knots is None else int(n_knots)
            if n < 2 or r_max <= 0:
                raise ValueError("n_knots must be >= 2 and r_max > 0")
            knots = np.linspace(0.0, float(r_max), n)
        self.knots = _validate_knots(knots)
        if state_knots is None:
            state_knots = np.linspace(0.0, 1.0, 4)
        self.state_knots = _validate_knots(state_knots)
        self.degree = int(degree)
        self.memory_depth = int(memory_depth)
        self.state_alphas = [float(a) for a in state_alphas]
        self.state_degree = int(state_degree)
        self.q_scale = float(q_scale)
        self.interaction = bool(interaction)
        self.coeffs: np.ndarray | None = None

    @classmethod
    def from_signal(cls, x: np.ndarray, n_knots: int = 8, degree: int = 3,
                    memory_depth: int = 4,
                    state_alphas: Sequence[float] = (0.99, 0.9999),
                    n_state_knots: int = 4, state_degree: int = 2,
                    interaction: bool = True, placement: str = "hybrid",
                    headroom: float = 1.05) -> StateConditionedSpline:
        """Resolve amplitude knots and the state normalization from a
        calibration signal (which should exercise the power dynamics —
        bursts/steps, not just one stationary capture)."""
        x = np.asarray(x)
        amps = np.abs(x)
        r_max = float(headroom * amps.max())
        ks = place_knots(amps, n_knots=n_knots, placement=placement,
                         r_max=r_max)
        q_scale = 1.0
        obj = cls(knots=ks, degree=degree, memory_depth=memory_depth,
                  state_alphas=state_alphas,
                  state_knots=np.linspace(0.0, 1.0, max(2, n_state_knots)),
                  state_degree=state_degree, q_scale=q_scale,
                  interaction=interaction)
        q_peak = max(float(q.max()) for q in obj._raw_states(x))
        obj.q_scale = float(headroom * max(q_peak, 1e-30))
        return obj

    def get_config(self) -> dict:
        return {"knots": [float(k) for k in self.knots],
                "degree": self.degree, "memory_depth": self.memory_depth,
                "state_alphas": [float(a) for a in self.state_alphas],
                "state_knots": [float(k) for k in self.state_knots],
                "state_degree": self.state_degree,
                "q_scale": self.q_scale,
                "interaction": self.interaction}

    # ---- dimensions --------------------------------------------------
    @property
    def n_basis(self) -> int:
        return len(self.knots) - 1 + self.degree

    @property
    def n_state_basis(self) -> int:
        return len(self.state_knots) - 1 + self.state_degree

    @property
    def n_states(self) -> int:
        return len(self.state_alphas)

    @property
    def n_coeffs(self) -> int:
        n = self.memory_depth * self.n_basis                  # baseline SMP
        n += self.n_states * self.memory_depth * self.n_state_basis
        if self.interaction:                                  # tap-0 surface
            n += self.n_states * self.n_basis * self.n_state_basis
        return n

    # ---- state extraction -------------------------------------------
    def _raw_states(self, x: np.ndarray) -> list[np.ndarray]:
        from scipy.signal import lfilter
        p = np.abs(np.asarray(x)) ** 2
        return [lfilter([1.0 - a], [1.0, -a], p)
                for a in self.state_alphas]

    def states(self, x: np.ndarray) -> list[np.ndarray]:
        """Normalized slow power states, clipped to the unit interval."""
        return [np.clip(q / self.q_scale, 0.0, 1.0)
                for q in self._raw_states(x)]

    # ---- basis -------------------------------------------------------
    def basis_matrix(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=complex)
        blocks = []
        for m in range(self.memory_depth):                    # baseline SMP
            xm = delayed(x, m)
            blocks.append(xm[:, None] * bspline_design_matrix(
                np.abs(xm), self.knots, self.degree))
        state_bases = [bspline_design_matrix(q, self.state_knots,
                                             self.state_degree)
                       for q in self.states(x)]
        for c in state_bases:                                 # additive gain
            for m in range(self.memory_depth):
                blocks.append(delayed(x, m)[:, None] * c)
        if self.interaction:                                  # tap-0 surface
            b0 = bspline_design_matrix(np.abs(x), self.knots, self.degree)
            for c in state_bases:
                tensor = (b0[:, :, None] * c[:, None, :]).reshape(len(x),
                                                                  -1)
                blocks.append(x[:, None] * tensor)
        return np.concatenate(blocks, axis=1)

    def passthrough_coeffs(self) -> np.ndarray:
        """Identity DPD: all-ones over the baseline tap-0 spline block."""
        w = np.zeros(self.n_coeffs, dtype=complex)
        w[:self.n_basis] = 1.0
        return w

    def smoothness_penalty(self) -> np.ndarray:
        """P-spline roughness over every J- or Js-sized coefficient block."""
        js, jd = self.n_basis, self.n_state_basis
        sizes = [js] * self.memory_depth
        sizes += [jd] * (self.n_states * self.memory_depth)
        if self.interaction:
            sizes += [jd] * (self.n_states * js)   # roughness along state axis
        rows = sum(max(s - 2, 0) for s in sizes)
        out = np.zeros((rows, self.n_coeffs))
        r0 = c0 = 0
        for s in sizes:
            d = _second_difference(s)
            out[r0:r0 + d.shape[0], c0:c0 + s] = d
            r0 += d.shape[0]
            c0 += s
        return out

    def fit(self, x: np.ndarray, y: np.ndarray,
            regularization: float = 0.0, smoothness: float = 0.0,
            weights: np.ndarray | None = None) -> StateConditionedSpline:
        self.coeffs = lstsq_fit(
            self.basis_matrix(x), y, regularization, weights=weights,
            penalty=self.smoothness_penalty() if smoothness > 0 else None,
            penalty_weight=smoothness)
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.coeffs is None:
            raise RuntimeError("model is not fitted; call fit(x, y) first")
        return self.basis_matrix(x) @ self.coeffs


class CoefficientScheduler:
    """Interpolate fitted coefficient vectors across an operating scalar.

    Fit the *same* model structure at several operating points
    (temperature, bias, average power), then evaluate anywhere in the
    covered range without retraining:

        sched = CoefficientScheduler.fit_conditions(
            lambda: SplineMemoryPolynomial(knots=ks), captures)
        model_35c = sched.at(35.0)

    PCHIP (default) is shape-preserving between calibration points;
    ``kind="linear"`` is the hardware-style two-point blend. Queries are
    clamped to the calibrated range (no extrapolation).
    """

    def __init__(self, models: Sequence[PAModel],
                 conditions: Sequence[float], kind: str = "pchip"):
        if len(models) != len(conditions):
            raise ValueError("one model per condition")
        if len(models) < 2:
            raise ValueError("need at least 2 operating points")
        if kind not in ("pchip", "linear"):
            raise ValueError("kind must be pchip|linear")
        cfg0 = models[0].get_config()
        if any(type(m) is not type(models[0]) or m.get_config() != cfg0
               for m in models[1:]):
            raise ValueError("all models must share class and config")
        if any(getattr(m, "coeffs", None) is None for m in models):
            raise ValueError("all models must be fitted")
        order = np.argsort(np.asarray(conditions, dtype=float))
        self.conditions = np.asarray(conditions, dtype=float)[order]
        if len(np.unique(self.conditions)) != len(self.conditions):
            raise ValueError("conditions must be distinct")
        self.coeff_table = np.stack(
            [np.asarray(models[i].coeffs, dtype=complex) for i in order])
        self.model_class = type(models[0])
        self.config = cfg0
        self.kind = kind

    @classmethod
    def fit_conditions(cls, model_factory,
                       captures: Sequence[tuple],
                       kind: str = "pchip", **fit_kwargs
                       ) -> CoefficientScheduler:
        """Fit ``model_factory()`` on each ``(condition, x, y)`` capture."""
        models, conds = [], []
        for cond, x, y in captures:
            models.append(model_factory().fit(x, y, **fit_kwargs))
            conds.append(float(cond))
        return cls(models, conds, kind=kind)

    def coeffs_at(self, condition: float) -> np.ndarray:
        c = float(np.clip(condition, self.conditions[0],
                          self.conditions[-1]))
        if self.kind == "linear" or len(self.conditions) == 2:
            re = np.array([np.interp(c, self.conditions, col.real)
                           for col in self.coeff_table.T])
            im = np.array([np.interp(c, self.conditions, col.imag)
                           for col in self.coeff_table.T])
            return re + 1j * im
        from scipy.interpolate import PchipInterpolator
        re = PchipInterpolator(self.conditions, self.coeff_table.real,
                               axis=0)(c)
        im = PchipInterpolator(self.conditions, self.coeff_table.imag,
                               axis=0)(c)
        return re + 1j * im

    def at(self, condition: float) -> PAModel:
        """Instantiate the scheduled model at an operating point."""
        m = self.model_class(**self.config)
        m.coeffs = self.coeffs_at(condition)
        return m

    def save(self, path: str) -> None:
        np.savez(path, scheduler=np.bytes_(b"CoefficientScheduler"),
                 model_class=self.model_class.__name__,
                 config=repr(self.config), kind=self.kind,
                 conditions=self.conditions, coeff_table=self.coeff_table)

    @classmethod
    def load(cls, path: str) -> CoefficientScheduler:
        import ast

        from . import _MODEL_CLASSES
        d = np.load(path, allow_pickle=False)
        model_class = _MODEL_CLASSES[str(d["model_class"])]
        config = ast.literal_eval(str(d["config"]))
        obj = cls.__new__(cls)
        obj.conditions = np.asarray(d["conditions"], dtype=float)
        obj.coeff_table = np.asarray(d["coeff_table"], dtype=complex)
        obj.model_class = model_class
        obj.config = config
        obj.kind = str(d["kind"])
        return obj
