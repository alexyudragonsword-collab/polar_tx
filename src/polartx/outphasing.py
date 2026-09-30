"""Outphasing (LINC / Chireix) transmitter, a sibling of the polar chain.

The polar chain puts the envelope on the DPA amplitude code and the phase
on one phase modulator.  Outphasing puts BOTH on phase: the signal

    x(t) = A(t) e^{jφ(t)},   θ(t) = arccos(A(t) / A_max)

is the sum of two constant-envelope branches

    s1 = (A_max/2) e^{j(φ+θ)},   s2 = (A_max/2) e^{j(φ-θ)},   s1 + s2 = x,

each of which is amplified by a PA running at constant full drive and then
vector-summed in a combiner.  The nonlinearity of the PAs never sees an
envelope, so it cannot distort one; what outphasing pays instead is power
in the combiner (θ far from 0 is power the load does not take) and an
acute sensitivity to phase mismatch between the branches.

This module follows the precedent of ``fir.py``: ``OutphasingTX`` wraps a
``PolarTX`` and runs it twice — once per branch — so the phase modulator,
the DPA (at its full code), the LO and every random impairment are the
ordinary chain's; ``OutphasingResult`` reuses ``PolarResult``'s metric
methods through ``_as_polar``.  ``chain.py`` and ``ChainConfig`` are not
touched, and the topology is selected by preset (``presets.wifi_outphasing``),
exactly as the dual-tap FIR chain is.

What the wrapped chain's ``ChainConfig`` means here:

- ``cfr_papr_db`` is applied ONCE to the composite signal before the
  decomposition — the same CFR the polar chain applies before its split, so
  the two topologies see the identical waveform.
- ``env_skew_s``, ``env_floor``, ``env_headroom``, ``fs_scale_fixed`` and
  the polar DPD are envelope-path knobs; a constant-envelope branch has no
  envelope path, so they are neutralized for the branch runs (``env_skew_s``
  is asserted to have no effect in the tests).  ``fs_scale_fixed`` still
  pins A_max when set, so a Monte-Carlo run stays a static system.
- ``phase_slew_max_hz``, ``supply``, ``f_dpa`` / ``interleave`` and the
  post-DPA memory model pass through unchanged: those are phase-path or
  per-PA properties the branches genuinely have.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .chain import PolarResult, PolarTX
from .dpa.combiner import OutphasingCombiner
from .vendor.padpd.cfr import cfr_clip_filter
from .waveforms.base import Waveform


def outphasing_decompose(x: np.ndarray, a_max: float
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a complex baseband signal into its two constant-envelope
    outphasing branches.

    Returns ``(s1, s2, theta)`` with ``s1 + s2 == x`` wherever ``|x| <= a_max``
    (samples above ``a_max`` are clipped to it, θ = 0).  θ is the outphasing
    half-angle in [0, π/2]: 0 at the peak, π/2 at a zero crossing, where the
    two branches cancel.
    """
    x = np.asarray(x, dtype=complex)
    a_max = float(a_max)
    if a_max <= 0.0:
        raise ValueError("a_max must be positive")
    ratio = np.clip(np.abs(x) / a_max, 0.0, 1.0)
    theta = np.arccos(ratio)
    phi = np.angle(x)
    half = 0.5 * a_max
    s1 = half * np.exp(1j * (phi + theta))
    s2 = half * np.exp(1j * (phi - theta))
    return s1, s2, theta


def theta_stats(theta: np.ndarray) -> dict:
    """Distribution of the outphasing angle [deg]: percentiles, maximum and
    the fraction of samples beyond 80° — where the branches are within 10°
    of cancelling, the region most sensitive to branch mismatch and where
    the isolated combiner burns almost everything."""
    deg = np.rad2deg(np.asarray(theta, float))
    return {"p50_deg": float(np.percentile(deg, 50)),
            "p95_deg": float(np.percentile(deg, 95)),
            "max_deg": float(deg.max()),
            "frac_gt80": float(np.mean(deg > 80.0))}


def phase_mismatch_evm_budget_db(x: np.ndarray, s2: np.ndarray,
                                 delta_deg: float, fs: float, bw: float
                                 ) -> float:
    """Closed-form EVM budget for a static phase error δ on branch 2.

    With branch 2 rotated by δ the combined output is ``x + (e^{jδ} - 1) s2``:
    the error vector is known exactly from the decomposition alone, before
    any PA or phase modulator is involved.  The budget scores it the way the
    chain's scalar-equalized OFDM EVM does — in-band (|f| <= bw/2, where the
    demodulated tones live) and after removing the complex gain a scalar
    equalizer would absorb — and returns 20 log10 of the rms ratio, the
    convention of ``analysis.responses``.  Small-angle intuition: the error
    magnitude is ``δ · A_max / 2`` per sample, so the budget scales with
    ``δ · sqrt(PAPR)``: outphasing's mismatch sensitivity grows with crest
    factor, which is the number to compare against the polar chain's
    ``env_skew_s`` sensitivity.
    """
    x = np.asarray(x, complex)
    err = (np.exp(1j * np.deg2rad(delta_deg)) - 1.0) * np.asarray(s2, complex)
    freq = np.fft.fftfreq(x.size, d=1.0 / fs)
    band = np.abs(freq) <= bw / 2.0
    xb = np.fft.ifft(np.where(band, np.fft.fft(x), 0.0))
    eb = np.fft.ifft(np.where(band, np.fft.fft(err), 0.0))
    g = np.vdot(xb, eb) / np.vdot(xb, xb)          # part a scalar EQ absorbs
    resid = eb - g * xb
    return float(20.0 * np.log10(
        np.sqrt(np.mean(np.abs(resid) ** 2) / np.mean(np.abs(xb) ** 2))))


@dataclass
class OutphasingResult:
    """One outphasing run: the combined output, the angle trajectory and BOTH
    branch runs intact.

    ``taps`` keeps each branch's ``PolarResult`` exactly as a single-chain
    run would return it (their ``env_code`` is the constant full code, their
    ``phase_out`` the modulated φ±θ), so a branch can be inspected on its
    own.  ``theta`` is the decomposition's outphasing angle [rad] on the
    baseband grid — ``theta_stats()`` summarizes it, and ``frac_gt80`` is
    the single number that says how hard this waveform is for outphasing.

    Scored with the SAME scalar equalization as ``PolarResult`` (the
    combiner is instantaneous, imposing no group delay), so the polar and
    outphasing numbers are directly comparable.
    """

    y: np.ndarray
    fs: float
    wf: Waveform
    theta: np.ndarray                  # outphasing angle [rad], per sample
    taps: tuple                        # (PolarResult branch 1, PolarResult branch 2)
    combiner: OutphasingCombiner
    info: dict = field(default_factory=dict)

    #: same convention as PolarResult — no linear response to equalize away
    evm_equalize_default = "scalar"

    def _as_polar(self) -> PolarResult:
        """Branch 1's PolarResult carrying the COMBINED output, so the
        ordinary metric layer scores the combined signal.  Envelope taps
        (``env_cmd``/``env_code``) are the constant full code and must not
        be read as the composite envelope — use ``theta`` for that."""
        return replace(self.taps[0], y=self.y)

    def evm(self, equalize: str = "scalar", **kw):
        """Constellation EVM of the combined output (scalar-equalized by
        default, like the polar chain)."""
        return self._as_polar().evm(equalize=equalize, **kw)

    def aclr(self, *a, **kw):
        """ACLR of the combined output."""
        return self._as_polar().aclr(*a, **kw)

    def psd(self, *a, **kw):
        """Welch PSD of the combined output."""
        return self._as_polar().psd(*a, **kw)

    def check_mask(self, *a, **kw):
        """Spectral-mask check on the combined output."""
        return self._as_polar().check_mask(*a, **kw)

    def avg_efficiency(self, dpa) -> dict:
        """Modulated average efficiency of the two-PA outphasing stage.

        The branch PAs sit at one constant code, so their efficiency is the
        DPA's own law evaluated there — the same ``eff`` law the polar chain
        is scored with, which is what makes the two topologies' numbers
        comparable.  The combiner then applies its power factor over the θ
        trajectory (``OutphasingCombiner.average_efficiency``)."""
        eta_pa = float(np.mean(dpa.efficiency(self.taps[0].env_code)))
        return self.combiner.average_efficiency(self.theta, eta_pa)

    def theta_stats(self) -> dict:
        """Percentiles / max of θ [deg] and the fraction beyond 80°."""
        return theta_stats(self.theta)


class OutphasingTX:
    """Outphasing transmitter wrapping a base ``PolarTX``.

    ``run()`` applies the base chain's CFR to the composite, decomposes it
    into two constant-envelope branches (``outphasing_decompose``), runs the
    base chain once per branch with independent noise (``seed`` and
    ``seed + 1``, as ``FIRDualTapTX`` does) and combines the two outputs
    through ``combiner``.  The branch runs use the base chain's phase
    modulator and DPA, with the envelope-path knobs neutralized (see the
    module docstring); ``branch_tx`` is that derived chain, kept public so
    it can be inspected.
    """

    def __init__(self, tx: PolarTX, combiner: OutphasingCombiner | None = None):
        self.tx = tx
        self.combiner = combiner if combiner is not None else OutphasingCombiner()
        # A constant-envelope branch has no envelope path: CFR was applied
        # to the composite, and hole punching, AM/PM skew, headroom, the
        # pinned full scale and the polar DPD are all envelope-domain.  The
        # phase path and per-PA properties pass through untouched.
        self.branch_tx = PolarTX(
            replace(tx.cfg, cfr_papr_db=None, env_floor=0.0, env_skew_s=0.0,
                    env_headroom=1.0, fs_scale_fixed=None),
            tx.phasemod, tx.dpa, dpd=None, memory=tx.memory)

    def a_max(self, x: np.ndarray) -> float:
        """Full-scale amplitude the decomposition is referenced to: the
        pinned ``fs_scale_fixed`` when the base chain sets one (a static
        system across runs), else this burst's peak."""
        fixed = self.tx.cfg.fs_scale_fixed
        return float(fixed) if fixed is not None else float(np.abs(x).max())

    def run(self, wf: Waveform, *, noise: bool = True, seed: int = 0
            ) -> OutphasingResult:
        """Run both branches and combine them.

        Branch 1 uses ``seed``, branch 2 ``seed + 1``: their deterministic
        content is the decomposition, their random noise (LO, jitter, DTC
        dither) is independent, as it is in two physical branches.
        """
        c = self.tx.cfg
        info: dict = {}
        x = wf.x
        if c.cfr_papr_db is not None:
            x = cfr_clip_filter(x, c.cfr_papr_db, wf.fs, wf.bw)
            info["cfr_papr_db"] = c.cfr_papr_db
        a_max = self.a_max(x)
        s1, s2, theta = outphasing_decompose(x, a_max)
        r1 = self.branch_tx.run(replace(wf, x=s1), noise=noise, seed=seed)
        r2 = self.branch_tx.run(replace(wf, x=s2), noise=noise, seed=seed + 1)
        y = self.combiner.combine(r1.y, r2.y)
        info["a_max"] = a_max
        info["theta"] = theta_stats(theta)
        info["combining_loss_db"] = self.combiner.combining_loss_db()
        info["phasemod"] = (r1.info.get("phasemod"), r2.info.get("phasemod"))
        return OutphasingResult(y=y, fs=wf.fs, wf=wf, theta=theta,
                                taps=(r1, r2), combiner=self.combiner,
                                info=info)
