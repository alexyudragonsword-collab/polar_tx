# Vendored from PA_DPD@44cbcb3: src/padpd/pa/spline.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Piecewise B-spline models: spline memory polynomial (SMP).

Replaces the global polynomial envelope basis |x|^k of MP/GMP with
locally-supported B-splines of the instantaneous amplitude:

    y(n) = sum_m sum_j  theta_{m,j} * x(n-m) * B_j(|x(n-m)|)

(the SMP structure of Pascual Campo et al., "Gradient-Adaptive
Spline-Interpolated LUT Methods for Low-Complexity Digital
Predistortion", IEEE TVT 2020). Compared to a polynomial basis:

- **Conditioning**: powers r, r^3, r^5, r^7 are strongly collinear over
  a finite amplitude range (condition numbers 1e7+); B-spline columns
  overlap only with their neighbours, so the LS problem is orders of
  magnitude better conditioned.
- **Runtime**: only ``degree+1`` basis functions are non-zero at any
  amplitude (4 for cubic), independent of the knot count — the natural
  software analogue of a hardware LUT + interpolation datapath
  (see :mod:`padpd.deploy.lut`).
- **Local control**: knot density concentrates parameters where the PA
  actually curves (compression knee) instead of raising a global order.

The model is linear in its coefficients for *fixed* knots, so it plugs
into the existing LS / ILA / AdaptiveDPD machinery unchanged. Knots are
placed from data only in :meth:`SplineMemoryPolynomial.from_signal`;
constructed instances always carry an explicit, fixed knot list (which
is what persistence and adaptive templates require).

A spline-Hammerstein (one shared spline LUT + FIR) is intentionally not
a separate class: it is the rank-1 constraint ``theta_{m,j} = h_m *
theta_j`` of the SMP — bilinear, needing alternating LS — while the SMP
is one closed-form solve and strictly more expressive. Its hardware
benefit (a single shared table) is delivered by LUT extraction instead.

Amplitudes beyond the last knot are clamped, i.e. the complex gain
saturates at its endpoint value — the safe extrapolation for rare
waveform peaks (an unconstrained cubic tail can send a predistorter to
huge drive values).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np

from .base import PAModel, lstsq_fit
from .memory_polynomial import delayed


def bspline_design_matrix(r: np.ndarray, breakpoints: Sequence[float],
                          degree: int = 3) -> np.ndarray:
    """Dense B-spline basis matrix ``(len(r), J)`` on a clamped knot vector.

    ``breakpoints`` are the K+1 strictly-ascending spline breakpoints
    (both endpoints included); ``J = K + degree``. ``r`` is clipped to
    ``[breakpoints[0], breakpoints[-1]]`` first, which realises
    hold-endpoint extrapolation. Vectorized Cox–de Boor recursion.
    """
    b = np.asarray(breakpoints, dtype=float)
    t = np.concatenate([np.full(degree, b[0]), b, np.full(degree, b[-1])])
    r = np.clip(np.asarray(r, dtype=float).ravel(), b[0], b[-1])
    n_k = len(t)
    # degree 0: half-open span indicators, right endpoint closed
    basis = np.zeros((r.size, n_k - 1))
    for i in range(n_k - 1):
        if t[i] < t[i + 1]:
            basis[:, i] = (r >= t[i]) & (r < t[i + 1])
    basis[r == t[-1], n_k - degree - 2] = 1.0
    for p in range(1, degree + 1):
        nxt = np.zeros((r.size, n_k - 1 - p))
        for i in range(n_k - 1 - p):
            d1 = t[i + p] - t[i]
            d2 = t[i + p + 1] - t[i + 1]
            col = np.zeros(r.size)
            if d1 > 0:
                col = (r - t[i]) / d1 * basis[:, i]
            if d2 > 0:
                col = col + (t[i + p + 1] - r) / d2 * basis[:, i + 1]
            nxt[:, i] = col
        basis = nxt
    return basis


def place_knots(amps: np.ndarray, n_knots: int = 8,
                placement: str = "hybrid", r_max: float | None = None,
                tail_start: float = 0.95,
                knee: float | None = None) -> list[float]:
    """Choose ``n_knots`` breakpoints on [0, r_max] from an amplitude sample.

    - ``"uniform"``: evenly spaced (the fixed-point-friendly baseline).
    - ``"quantile"``: empirical amplitude quantiles — every span gets
      comparable sample occupancy, but the sparse peak tail is starved.
    - ``"hybrid"`` (default): quantiles below the ``tail_start`` quantile
      plus a uniform tail up to ``r_max`` — resolution where the data
      lives *and* guaranteed knots in the compression/peak region.

    ``knee`` (e.g. an estimated 1-dB compression amplitude) replaces the
    nearest interior knot so the count stays ``n_knots``.
    """
    amps = np.abs(np.asarray(amps, dtype=float)).ravel()
    if n_knots < 2:
        raise ValueError("n_knots must be >= 2")
    if r_max is None:
        if amps.size == 0:
            raise ValueError("r_max is required when amps is empty")
        r_max = float(amps.max())
    if r_max <= 0:
        raise ValueError("r_max must be > 0")

    if placement == "uniform":
        ks = np.linspace(0.0, r_max, n_knots)
    elif placement == "quantile":
        levels = np.linspace(0.0, 1.0, n_knots)[1:-1]
        ks = np.concatenate([[0.0], np.quantile(amps, levels), [r_max]])
    elif placement == "hybrid":
        n_tail = min(max(2, int(np.ceil(n_knots * (1 - tail_start))) + 1),
                     n_knots - 1)
        n_head = n_knots - n_tail
        q_edge = float(np.quantile(amps, tail_start))
        if not 0.0 < q_edge < r_max:      # degenerate stats -> uniform
            ks = np.linspace(0.0, r_max, n_knots)
        else:
            levels = np.linspace(0.0, tail_start, n_head + 1)[1:-1]
            head = np.concatenate([[0.0], np.quantile(amps, levels)])
            ks = np.concatenate([head, np.linspace(q_edge, r_max, n_tail)])
    else:
        raise ValueError("placement must be uniform|quantile|hybrid")

    if knee is not None and 0.0 < knee < r_max:
        interior = np.arange(1, len(ks) - 1)
        ks[interior[np.argmin(np.abs(ks[interior] - knee))]] = knee
    ks = np.unique(ks)
    if len(ks) < 2:
        raise ValueError("degenerate amplitude data: knots collapsed")
    return [float(k) for k in ks]


def _second_difference(j: int) -> np.ndarray:
    """(j-2, j) second-difference operator over spline control points."""
    if j < 3:
        return np.zeros((0, j))
    d = np.zeros((j - 2, j))
    idx = np.arange(j - 2)
    d[idx, idx] = 1.0
    d[idx, idx + 1] = -2.0
    d[idx, idx + 2] = 1.0
    return d


def _phase_carrier(x: np.ndarray, order: int) -> np.ndarray:
    """Carrier of a given phase-harmonic order: +1 -> x, -1 -> conj(x)
    (image), -3 -> conj(x)^3 (counter-IM3). Each order rotates the
    envelope phase differently (exp(j*order*phi)), so the classes are
    mutually irreplaceable basis families."""
    if order == 1:
        return x
    if order == -1:
        return np.conj(x)
    if order == -3:
        return np.conj(x) ** 3
    raise ValueError(f"unsupported phase order {order}")


def _validate_knots(knots: Sequence[float]) -> list[float]:
    ks = [float(k) for k in knots]
    if len(ks) < 2:
        raise ValueError("need at least 2 knots (breakpoints)")
    if ks[0] < 0:
        raise ValueError("knots must be >= 0 (amplitude axis)")
    if any(b <= a for a, b in pairwise(ks)):
        raise ValueError("knots must be strictly ascending")
    return ks


class SplineMemoryPolynomial(PAModel):
    """Spline memory polynomial: x(n-m) * B_j(|x(n-m)|), linear in theta.

    Works both as a DPD basis (ILA / AdaptiveDPD) and as a PA forward
    behavioral model; ``memory_depth=1`` is a fittable complex-gain
    spline static nonlinearity (the trainable generalization of the
    ``WienerHammersteinPA`` read-only LUT). ``degree=1`` yields a
    linearly-interpolated LUT model directly.

    ``conjugate=True`` appends widely-linear image branches
    ``conj(x(n-m)) * B_j(|x(n-m)|)`` — the spline generalization of the
    x* and x*|x|^2 terms that model image-frequency distortion from TX
    I/Q imbalance (a phase-equivariant basis cannot represent an image,
    which rotates opposite to the carrier). The conjugate blocks are
    linearly independent of the direct ones (x and x* are independent
    complex directions), so no identifiability correction is needed.

    ``cim3=True`` appends counter-IM3 branches
    ``conj(x(n-m))^3 * B_j(|x(n-m)|)`` for the LO-3BB product of a
    direct-conversion TX (mixer 3rd-LO-harmonic path, and PA IM3 of
    image x wanted — both land on the same conj^3 term). This is the
    exp(-j3*phi) phase-harmonic class: conj(x)*f(|x|) rotates exp(-j*phi)
    only, so no conjugate-branch gain function can absorb it.
    ``dc_term=True`` adds one constant column for LO leakage.
    """

    def __init__(self, knots: Sequence[float] | None = None,
                 degree: int = 3, memory_depth: int = 4,
                 conjugate: bool = False, cim3: bool = False,
                 dc_term: bool = False,
                 n_knots: int | None = None, r_max: float = 1.0):
        if degree not in (1, 2, 3):
            raise ValueError("degree must be 1, 2 or 3")
        if memory_depth < 1:
            raise ValueError("memory_depth must be >= 1")
        if knots is None:
            n = 8 if n_knots is None else int(n_knots)
            if n < 2 or r_max <= 0:
                raise ValueError("n_knots must be >= 2 and r_max > 0")
            knots = np.linspace(0.0, float(r_max), n)
        self.knots = _validate_knots(knots)
        self.degree = int(degree)
        self.memory_depth = int(memory_depth)
        self.conjugate = bool(conjugate)
        self.cim3 = bool(cim3)
        self.dc_term = bool(dc_term)
        self.coeffs: np.ndarray | None = None

    @classmethod
    def from_signal(cls, x: np.ndarray, n_knots: int = 8, degree: int = 3,
                    memory_depth: int = 4, conjugate: bool = False,
                    cim3: bool = False, dc_term: bool = False,
                    placement: str = "hybrid",
                    headroom: float = 1.05) -> SplineMemoryPolynomial:
        """Resolve data-driven knots from a calibration signal, then build.

        ``headroom`` scales the top knot beyond the observed peak so the
        clamp region only handles genuinely unseen amplitudes.
        """
        amps = np.abs(np.asarray(x))
        r_max = float(headroom * amps.max())
        ks = place_knots(amps, n_knots=n_knots, placement=placement,
                         r_max=r_max)
        return cls(knots=ks, degree=degree, memory_depth=memory_depth,
                   conjugate=conjugate, cim3=cim3, dc_term=dc_term)

    def get_config(self) -> dict:
        return {"knots": [float(k) for k in self.knots],
                "degree": self.degree, "memory_depth": self.memory_depth,
                "conjugate": self.conjugate, "cim3": self.cim3,
                "dc_term": self.dc_term}

    @property
    def n_basis(self) -> int:
        """Basis functions per memory tap: J = (#breakpoints - 1) + degree."""
        return len(self.knots) - 1 + self.degree

    @property
    def n_branches(self) -> int:
        return self.memory_depth * (1 + self.conjugate + self.cim3)

    @property
    def n_coeffs(self) -> int:
        return self.n_branches * self.n_basis + (1 if self.dc_term else 0)

    def basis_matrix(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=complex)
        blocks = []
        for order in self.branch_phase_orders()[::self.memory_depth]:
            for m in range(self.memory_depth):
                xm = delayed(x, m)
                b = bspline_design_matrix(np.abs(xm), self.knots,
                                          self.degree)
                blocks.append(_phase_carrier(xm, order)[:, None] * b)
        if self.dc_term:
            blocks.append(np.ones((len(x), 1), dtype=complex))
        return np.concatenate(blocks, axis=1)

    def passthrough_coeffs(self) -> np.ndarray:
        """Coefficients realising DPD(x) = x exactly.

        B-splines sum to one (partition of unity), so all-ones over the
        tap-0 block is the identity — the spline analogue of ``w[0]=1``
        on a polynomial basis. Used by AdaptiveDPD's pass-through init.
        """
        w = np.zeros(self.n_coeffs, dtype=complex)
        w[:self.n_basis] = 1.0
        return w

    def smoothness_penalty(self) -> np.ndarray:
        """Block-diagonal P-spline roughness operator (one D2 per branch;
        the DC column, if any, carries no roughness)."""
        p = np.kron(np.eye(self.n_branches),
                    _second_difference(self.n_basis))
        if self.dc_term:
            p = np.hstack([p, np.zeros((p.shape[0], 1))])
        return p

    def branch_delays(self) -> list[tuple[int, int]]:
        """(carrier delay, envelope delay) per branch — LUT metadata."""
        taps = [(m, m) for m in range(self.memory_depth)]
        return taps * (1 + self.conjugate + self.cim3)

    def branch_phase_orders(self) -> list[int]:
        """Phase-harmonic order of each branch's carrier: +1 for x, -1
        for conj(x) (image), -3 for conj(x)^3 (counter-IM3)."""
        orders = [1] * self.memory_depth
        if self.conjugate:
            orders += [-1] * self.memory_depth
        if self.cim3:
            orders += [-3] * self.memory_depth
        return orders

    def branch_conjugate(self) -> list[bool]:
        """Whether each branch's carrier is phase-conjugated (order < 0)."""
        return [o < 0 for o in self.branch_phase_orders()]

    def dc_coefficient(self) -> complex | None:
        """Fitted constant (LO-leakage) term, if the model carries one."""
        if not self.dc_term or self.coeffs is None:
            return None
        return complex(self.coeffs[-1])

    def gain_curve(self, r: np.ndarray) -> np.ndarray:
        """Complex gain of each branch vs envelope: (n_branches, len(r)).

        Branch ``b`` contributes ``carrier_b(n) * gain_b(|x(n-e_b)|)``
        with carrier x, conj(x) or conj(x)^3 per
        :meth:`branch_phase_orders`; this is the curve a hardware LUT
        stores (see :mod:`padpd.deploy.lut`). The DC term, if any, is
        not a branch — read it via :meth:`dc_coefficient`.
        """
        if self.coeffs is None:
            raise RuntimeError("model is not fitted; call fit(x, y) first")
        b = bspline_design_matrix(np.asarray(r, dtype=float), self.knots,
                                  self.degree)
        j = self.n_basis
        return np.stack([b @ self.coeffs[k * j:(k + 1) * j]
                         for k in range(self.n_branches)])

    def fit(self, x: np.ndarray, y: np.ndarray,
            regularization: float = 0.0, smoothness: float = 0.0,
            weights: np.ndarray | None = None) -> SplineMemoryPolynomial:
        """LS fit; ``smoothness`` adds the P-spline second-difference
        penalty (relative, like ``regularization``), ``weights`` are
        per-sample WLS weights."""
        self.coeffs = lstsq_fit(
            self.basis_matrix(x), y, regularization, weights=weights,
            penalty=self.smoothness_penalty() if smoothness > 0 else None,
            penalty_weight=smoothness)
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.coeffs is None:
            raise RuntimeError("model is not fitted; call fit(x, y) first")
        return self.basis_matrix(x) @ self.coeffs


class SplineGMP(PAModel):
    """Spline GMP: adds lagging/leading cross-envelope spline branches.

    The spline analogue of :class:`~padpd.pa.gmp.GMPModel` (Morgan et
    al., 2006): the carrier sample and the envelope sample may carry
    different delays, which captures phase-vs-delayed-amplitude memory
    that aligned-envelope terms miss.

    Basis (shared knots; aligned branches expand into J spline columns):
      aligned : x(n-m) B_j(|x(n-m)|)      m=0..M-1
      lagging : x(n-m) B_j(|x(n-m-l)|)    m=0..Mb-1, l=1..Lb
      leading : x(n-m) B_j(|x(n-m+l)|)    m=0..Mc-1, l=1..Lc

    Cross (lag/lead) branches drop their first basis column (J-1
    columns each): by partition of unity every branch's full column
    block sums to its carrier x(n-m), so a cross branch sharing a
    carrier delay with an aligned branch would be *exactly* rank
    deficient. Removing one column removes that direction (the
    constant-gain component, already provided by the aligned branch)
    and keeps the LS problem well posed.
    """

    def __init__(self, knots: Sequence[float] | None = None,
                 degree: int = 3, memory_depth: int = 4,
                 lag_memory: int = 2, lag_span: int = 1,
                 lead_memory: int = 2, lead_span: int = 1,
                 n_knots: int | None = None, r_max: float = 1.0):
        if degree not in (1, 2, 3):
            raise ValueError("degree must be 1, 2 or 3")
        if memory_depth < 1:
            raise ValueError("memory_depth must be >= 1")
        if min(lag_memory, lag_span, lead_memory, lead_span) < 0:
            raise ValueError("lag/lead memories and spans must be >= 0")
        if knots is None:
            n = 8 if n_knots is None else int(n_knots)
            if n < 2 or r_max <= 0:
                raise ValueError("n_knots must be >= 2 and r_max > 0")
            knots = np.linspace(0.0, float(r_max), n)
        self.knots = _validate_knots(knots)
        self.degree = int(degree)
        self.memory_depth = int(memory_depth)
        self.lag_memory = int(lag_memory)
        self.lag_span = int(lag_span)
        self.lead_memory = int(lead_memory)
        self.lead_span = int(lead_span)
        self.coeffs: np.ndarray | None = None

    @classmethod
    def from_signal(cls, x: np.ndarray, n_knots: int = 8, degree: int = 3,
                    memory_depth: int = 4, lag_memory: int = 2,
                    lag_span: int = 1, lead_memory: int = 2,
                    lead_span: int = 1, placement: str = "hybrid",
                    headroom: float = 1.05) -> SplineGMP:
        amps = np.abs(np.asarray(x))
        r_max = float(headroom * amps.max())
        ks = place_knots(amps, n_knots=n_knots, placement=placement,
                         r_max=r_max)
        return cls(knots=ks, degree=degree, memory_depth=memory_depth,
                   lag_memory=lag_memory, lag_span=lag_span,
                   lead_memory=lead_memory, lead_span=lead_span)

    def get_config(self) -> dict:
        return {"knots": [float(k) for k in self.knots],
                "degree": self.degree, "memory_depth": self.memory_depth,
                "lag_memory": self.lag_memory, "lag_span": self.lag_span,
                "lead_memory": self.lead_memory,
                "lead_span": self.lead_span}

    @property
    def n_basis(self) -> int:
        return len(self.knots) - 1 + self.degree

    @property
    def n_branches(self) -> int:
        return (self.memory_depth + self.lag_memory * self.lag_span
                + self.lead_memory * self.lead_span)

    @property
    def n_cross_branches(self) -> int:
        return (self.lag_memory * self.lag_span
                + self.lead_memory * self.lead_span)

    def _branch_widths(self) -> list[int]:
        j = self.n_basis
        return ([j] * self.memory_depth
                + [j - 1] * self.n_cross_branches)

    @property
    def n_coeffs(self) -> int:
        return sum(self._branch_widths())

    def basis_matrix(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=complex)
        a = np.abs(x)
        blocks = []

        def block(xm: np.ndarray, env: np.ndarray,
                  cross: bool) -> np.ndarray:
            b = bspline_design_matrix(env, self.knots, self.degree)
            if cross:                    # identifiability: see class doc
                b = b[:, 1:]
            return xm[:, None] * b

        for m in range(self.memory_depth):
            blocks.append(block(delayed(x, m), delayed(a, m), False))
        for m in range(self.lag_memory):
            xm = delayed(x, m)
            for lag in range(1, self.lag_span + 1):
                blocks.append(block(xm, delayed(a, m + lag), True))
        for m in range(self.lead_memory):
            xm = delayed(x, m)
            for lead in range(1, self.lead_span + 1):
                blocks.append(block(xm, delayed(a, m - lead), True))
        return np.concatenate(blocks, axis=1)

    def passthrough_coeffs(self) -> np.ndarray:
        """Identity DPD: all-ones over the aligned tap-0 spline block."""
        w = np.zeros(self.n_coeffs, dtype=complex)
        w[:self.n_basis] = 1.0
        return w

    def smoothness_penalty(self) -> np.ndarray:
        """Block-diagonal P-spline roughness operator (one D2 per branch)."""
        widths = self._branch_widths()
        rows = sum(max(w - 2, 0) for w in widths)
        out = np.zeros((rows, self.n_coeffs))
        r0 = c0 = 0
        for w in widths:
            d = _second_difference(w)
            out[r0:r0 + d.shape[0], c0:c0 + w] = d
            r0 += d.shape[0]
            c0 += w
        return out

    def branch_delays(self) -> list[tuple[int, int]]:
        """(carrier delay, envelope delay) per branch, basis order."""
        pairs = [(m, m) for m in range(self.memory_depth)]
        pairs += [(m, m + lag) for m in range(self.lag_memory)
                  for lag in range(1, self.lag_span + 1)]
        pairs += [(m, m - lead) for m in range(self.lead_memory)
                  for lead in range(1, self.lead_span + 1)]
        return pairs

    def gain_curve(self, r: np.ndarray) -> np.ndarray:
        """Complex gain of each branch vs envelope: (n_branches, len(r))."""
        if self.coeffs is None:
            raise RuntimeError("model is not fitted; call fit(x, y) first")
        b = bspline_design_matrix(np.asarray(r, dtype=float), self.knots,
                                  self.degree)
        curves, c0 = [], 0
        for w in self._branch_widths():
            bb = b if w == self.n_basis else b[:, 1:]
            curves.append(bb @ self.coeffs[c0:c0 + w])
            c0 += w
        return np.stack(curves)

    def fit(self, x: np.ndarray, y: np.ndarray,
            regularization: float = 0.0, smoothness: float = 0.0,
            weights: np.ndarray | None = None) -> SplineGMP:
        self.coeffs = lstsq_fit(
            self.basis_matrix(x), y, regularization, weights=weights,
            penalty=self.smoothness_penalty() if smoothness > 0 else None,
            penalty_weight=smoothness)
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.coeffs is None:
            raise RuntimeError("model is not fitted; call fit(x, y) first")
        return self.basis_matrix(x) @ self.coeffs
