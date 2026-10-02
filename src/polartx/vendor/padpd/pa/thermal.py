# Vendored from PA_DPD@44cbcb3: src/padpd/pa/thermal.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Self-heating virtual DUT: dissipated power drives the drift state.

:class:`~padpd.pa.drift.DriftingReferencePA` exposes a drift state the
*caller* sets — good for aging studies, but thermal memory is not an
external knob: the PA heats itself as a function of its own recent
output power. :class:`ThermalReferencePA` closes that loop with a
multi-pole RC thermal network,

    theta_k <- a_k * theta_k + (1 - a_k) * w_k * P_diss ,
    state    = heat_gain * sum_k theta_k                (clipped to 0..1)

advanced block-wise (the state is frozen within a block — thermal time
constants are orders of magnitude slower than the sample rate), with
``P_diss`` proxied by the block's mean output power normalized to the
full-drive reference. The state feeds the same drive/knee/AM-PM drift
mapping as the drifting PA, so hot means more compression and more
phase distortion — the behavior a state-conditioned model must capture
and a memoryless-state model cannot.

Use :func:`burst_stimulus` to build power-stepped waveforms that
actually exercise the transient (a stationary capture never separates
thermal states from bias).
"""

from __future__ import annotations

import numpy as np

from .drift import DriftingReferencePA


def burst_stimulus(x: np.ndarray, n_bursts: int = 4,
                   low_scale: float = 0.35) -> np.ndarray:
    """Alternate full-power and backed-off segments of a waveform.

    Splits ``x`` into ``2 * n_bursts`` equal segments and scales every
    other one by ``low_scale`` — a heating/cooling exercise pattern for
    thermal identification.
    """
    if n_bursts < 1:
        raise ValueError("n_bursts must be >= 1")
    x = np.asarray(x, dtype=complex).copy()
    n_seg = 2 * n_bursts
    edges = np.linspace(0, len(x), n_seg + 1).astype(int)
    for s in range(1, n_seg, 2):
        x[edges[s]:edges[s + 1]] *= low_scale
    return x


class ThermalReferencePA:
    """ReferencePA with power-driven electrothermal drift (stateful).

    Not a fittable model — a virtual DUT. Call :meth:`reset` between
    independent captures; the thermal state otherwise persists across
    calls, exactly like a real device that stays warm.

    Defaults are tuned as a *pronounced* self-heating demo: thermal time
    constants (5 us / 30 us) that actually evolve within a ~150 us
    capture, drift spans stronger than DriftingReferencePA's, and
    ``heat_gain`` chosen so the state swings without pinning at 1.0
    (a saturated state is unobservable and unlearnable). Real GaN
    dynamics are slower — scale ``taus_s`` with your capture length.
    """

    def __init__(self, drive0: float = 0.13, drive_span: float = 0.05,
                 beta_a_span: float = 0.4, alpha_p_span: float = 1.2,
                 taus_s: tuple = (5e-6, 3e-5), weights: tuple = (0.6, 0.4),
                 heat_gain: float = 0.7, fs: float = 320e6,
                 block: int = 128):
        if len(taus_s) != len(weights):
            raise ValueError("one weight per thermal time constant")
        if min(taus_s) <= 0 or min(weights) < 0:
            raise ValueError("taus must be > 0 and weights >= 0")
        if block < 1:
            raise ValueError("block must be >= 1")
        self._drift = DriftingReferencePA(drive0=drive0,
                                          drive_span=drive_span,
                                          beta_a_span=beta_a_span,
                                          alpha_p_span=alpha_p_span)
        self.taus_s = tuple(float(t) for t in taus_s)
        self.weights = tuple(float(w) for w in weights)
        self.heat_gain = float(heat_gain)
        self.fs = float(fs)
        self.block = int(block)
        self._p_ref: float | None = None
        self.theta = np.zeros(len(taus_s))

    def reset(self) -> None:
        """Cold start: zero the thermal state (keeps the P_diss reference)."""
        self.theta = np.zeros(len(self.taus_s))

    @property
    def state(self) -> float:
        """Current drift state in [0, 1] (0 = cold, 1 = hot)."""
        return float(np.clip(self.heat_gain * self.theta.sum(), 0.0, 1.0))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=complex)
        y = np.empty_like(x)
        a = np.exp(-self.block / (np.asarray(self.taus_s) * self.fs))
        if self._p_ref is None and len(x):
            # reference dissipation: mean power of the first capture at the
            # current (cold) state — robust against a quiet leading block
            self._drift.set_state(self.state)
            self._p_ref = max(float(np.mean(
                np.abs(self._drift.pa()(x)) ** 2)), 1e-30)
        for i0 in range(0, len(x), self.block):
            chunk = x[i0:i0 + self.block]
            self._drift.set_state(self.state)
            yc = self._drift.pa()(chunk)
            y[i0:i0 + len(chunk)] = yc
            p_norm = float(np.mean(np.abs(yc) ** 2)) / self._p_ref
            self.theta = a * self.theta + (1 - a) * np.asarray(
                self.weights) * p_norm
        return y
