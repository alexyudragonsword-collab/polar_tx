# Vendored from PA_DPD@08b9725: src/padpd/pa/thermal.py
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

Only the *nonlinearity* is block-wise. The device is the same
Wiener-Hammerstein chain as :class:`~padpd.pa.reference_pa.ReferencePA`
(FIR_in -> Saleh -> FIR_out), and the drift mapping changes only the
Saleh stage and the drive: the two FIRs model matching networks, which do
not move with junction temperature and which see the signal as one
continuous waveform. So FIR_in runs over the whole capture, the Saleh
stage is applied block by block with that block's thermal parameters, and
FIR_out runs causally over the whole capture, carrying its tail across
block edges. With the state frozen this is exactly one ReferencePA call.

History: until 2026-10 each block was sent through a fresh ReferencePA,
whose FIRs convolve from zero on every call. Every block therefore lost
the previous block's tail, its first ``len(fir_in) - 1`` samples were
wrong, and the error depended only on (sample index mod block) — a
non-physical floor of -35.1 dB NMSE at the default block=128 (measured
with the heat frozen against one continuous call) that no causal model
could fit. Figures measured on this DUT before that date are lower
bounds. See tests/test_thermal_dut.py.

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

    def _transistor(self, u: np.ndarray) -> np.ndarray:
        """Saleh stage at the current drift state, gain-normalized.

        The 1/(drive * small_signal_gain) normalization that ReferencePA
        applies after FIR_out is applied here, before it: the drive moves
        with the thermal state, so each block's samples must carry their
        own block's normalization into the shared output FIR.
        """
        pa = self._drift.pa()
        return pa.saleh(pa.drive * u) / (pa.drive
                                         * pa.saleh.small_signal_gain)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """One continuous capture.

        The FIRs start from rest at the start of each call, as
        ReferencePA's do; the thermal state persists across calls. A
        waveform that is physically continuous should be passed in one
        call.
        """
        x = np.asarray(x, dtype=complex)
        n = len(x)
        y = np.empty_like(x)
        if n == 0:
            return y
        # the FIRs are matching networks: fixed, state-independent
        fir_pa = self._drift.pa()
        fir_out = fir_pa.fir_out
        n_tail = len(fir_out) - 1
        u = np.convolve(x, fir_pa.fir_in)[:n]
        if self._p_ref is None:
            # reference dissipation: mean power of the first capture at the
            # current (cold) state — robust against a quiet leading block.
            # Same signal path as the blocks below, state held fixed.
            self._drift.set_state(self.state)
            w = np.convolve(self._transistor(u), fir_out)[:n]
            self._p_ref = max(float(np.mean(np.abs(w) ** 2)), 1e-30)
        a = np.exp(-self.block / (np.asarray(self.taus_s) * self.fs))
        weights = np.asarray(self.weights)
        tail = np.zeros(n_tail, dtype=complex)   # FIR_out input history
        for i0 in range(0, n, self.block):
            self._drift.set_state(self.state)
            v = self._transistor(u[i0:i0 + self.block])
            buf = np.concatenate([tail, v])
            yc = np.convolve(buf, fir_out)[n_tail:n_tail + len(v)]
            tail = buf[len(buf) - n_tail:]
            y[i0:i0 + len(v)] = yc
            # P_diss proxy: this block's mean power after FIR_out, i.e. of
            # exactly the samples the device delivers for this block
            p_norm = float(np.mean(np.abs(yc) ** 2)) / self._p_ref
            self.theta = a * self.theta + (1 - a) * weights * p_norm
        return y
