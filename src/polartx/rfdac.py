"""RF-DAC: the digital I/Q transmitter's output stage.

Where the polar DPA takes an amplitude code and a modulated carrier, the
RF-DAC takes two signed codes — I and Q — and drives two unit-cell arrays
clocked by quadrature phases of a fixed LO.  There is no phase path at all:
the constellation is drawn directly in Cartesian coordinates, so nothing in
the chain ever sees an envelope, a phase trajectory or a bandwidth expansion.
What it pays for that is (a) the two arrays draw current in proportion to
|I| + |Q| while the load only takes I² + Q² — the diagonal of the code
square is ~1.5 dB less efficient than its axes — and (b) a set of
impairments that are specifically Cartesian: I/Q gain and phase imbalance
(an image), LO leakage (a carrier feedthrough) and LO clock jitter.

Shared with the polar DPA on purpose:

* the unit-cell mismatch model is the SAME ``dpa.mismatch.code_amplitude_table``
  with the same segmentation and seed semantics — an ``RFDAC`` built with a
  DPA's ``(n_bits, n_thermo, sigma_cell, gradient, seed)`` reproduces that
  DPA's INL/DNL exactly on its I array (Q draws with ``seed + 1``).  That is
  the definition of "same mismatch model" the comparison rests on, and it is
  pinned by a test;
* clock jitter is converted to phase the way the DTC modulator converts its
  edge jitter: a white draw of ``2π · fout · jitter_rms_s`` rad per sample,
  gated by ``noise``.

Sign-magnitude coding: ``n_bits`` is the MAGNITUDE resolution per axis, the
polarity is a differential steering switch, so an axis spans
``-(2^n_bits - 1) .. +(2^n_bits - 1)`` and full scale on an axis is 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dpa.mismatch import code_amplitude_table, inl_dnl
from .vendor.pllsim.blocks.oscillator import OscConfig
from .vendor.pllsim.core.colored import synth_from_psd

TWOPI = 2.0 * np.pi


def iq_image_rejection_db(gain_db: float, phase_deg: float) -> float:
    """Image rejection ratio [dB, positive] for a quadrature gain error
    ``gain_db`` and phase error ``phase_deg`` on the Q path.

    With ``y = I + j·K·Q``, ``K = (1 + g)·e^{jφ}``, the output is
    ``((1 + K)/2)·x + ((1 − K)/2)·x*``: signal and image gains, exactly.
    ``IRR = |1 + K|² / |1 − K|²``.  Small-angle: ``4 / (g² + φ²)``.
    """
    k = 10.0 ** (gain_db / 20.0) * np.exp(1j * np.deg2rad(phase_deg))
    num, den = abs(1.0 + k) ** 2, abs(1.0 - k) ** 2
    return float(10.0 * np.log10(num / max(den, 1e-300)))


@dataclass
class RFDACConfig:
    """Two-array I/Q current-steering DAC and its Cartesian impairments.

    n_bits / n_thermo / sigma_cell / gradient / seed
        Per-axis array, with exactly ``DPAConfig``'s meaning: magnitude
        bits, thermometer MSBs, relative unit mismatch, systematic tilt,
        RNG seed.  The I array draws with ``seed``, the Q array with
        ``seed + 1``.
    iq_gain_db / iq_phase_deg
        Quadrature gain and phase error applied to the Q path.  The image
        they raise is ``iq_image_rejection_db`` below the signal.
    lo_leakage_dbc
        Carrier feedthrough, as a DC term relative to an on-axis full-scale
        output; ``None`` = none.
    jitter_rms_s / fout
        LO clock jitter [s] and the RF output frequency it is scaled by,
        the DTC modulator's convention: white phase of ``2π·fout·σ_τ`` rad
        per sample, random, gated by ``noise``.
    lo_pn / lo_loop_bw
        The LO's Leeson phase noise (``OscConfig``) and the PLL loop
        bandwidth below which it is flattened — exactly ``DTCPMConfig``'s
        fields and generator, because the RF-DAC's carrier comes from the
        same kind of synthesizer and multiplies the output the same way.
        ``None`` = a noiseless LO (the stage-2 default, which made the
        RF-DAC look ~15 dB better than its LO allows; the ``wifi_rfdac``
        preset now carries the plan's LO).  Gated by ``noise``.
    eff
        ``("iq_cells", eta_peak)``: the load takes ``I² + Q²`` while the two
        arrays draw ``|I| + |Q|``, so ``η = η_peak · (I² + Q²) / (|I| + |Q|)``
        — ``η_peak`` on an axis at full scale, ``η_peak/√2`` on the diagonal.
        No charging-loss term, i.e. the optimistic RF-DAC law; it still
        sits below the polar SCPA law at backoff, which is the point.
    """

    n_bits: int = 12
    n_thermo: int = 6
    sigma_cell: float = 0.0
    gradient: float = 0.0
    seed: int = 0
    iq_gain_db: float = 0.0
    iq_phase_deg: float = 0.0
    lo_leakage_dbc: float | None = None
    jitter_rms_s: float = 0.0
    fout: float = 5.9e9
    lo_pn: OscConfig | None = None
    lo_loop_bw: float = 200e3
    eff: tuple = ("iq_cells", 0.85)

    @property
    def full_scale_code(self) -> int:
        """Largest magnitude code on an axis, ``2**n_bits - 1``."""
        return (1 << self.n_bits) - 1


class RFDAC:
    """A configured I/Q RF-DAC: two signed code streams -> complex baseband.

    ``amp_table_i`` / ``amp_table_q`` are the realized per-code amplitudes
    (normalized to the axis full scale) with the array mismatch folded in,
    public for the same reason the DPA's tables are.
    """

    def __init__(self, cfg: RFDACConfig):
        self.cfg = cfg
        nt = min(cfg.n_thermo, cfg.n_bits)
        self._raw_i = code_amplitude_table(cfg.n_bits, nt, cfg.sigma_cell,
                                           cfg.gradient,
                                           np.random.default_rng(cfg.seed))
        self._raw_q = code_amplitude_table(cfg.n_bits, nt, cfg.sigma_cell,
                                           cfg.gradient,
                                           np.random.default_rng(cfg.seed + 1))
        self.amp_table_i = self._raw_i / self._raw_i[-1]
        self.amp_table_q = self._raw_q / self._raw_q[-1]
        if cfg.eff[0] != "iq_cells":
            raise ValueError(f"unknown RF-DAC efficiency law {cfg.eff!r}")

    # ------------------------------------------------------------- codes
    def encode(self, v: np.ndarray) -> np.ndarray:
        """Normalized axis value in [-1, 1] -> signed magnitude code."""
        fs = self.cfg.full_scale_code
        return np.clip(np.rint(np.asarray(v, float) * fs), -fs, fs).astype(np.int64)

    def _amp(self, code: np.ndarray, table: np.ndarray) -> np.ndarray:
        code = np.asarray(code, np.int64)
        return np.sign(code) * table[np.abs(code)]

    def __call__(self, i_code: np.ndarray, q_code: np.ndarray, *,
                 noise: bool = True, seed: int = 0,
                 fs: float | None = None) -> np.ndarray:
        """Complex-baseband output for the two code streams, axis full
        scale = 1: mismatch, I/Q imbalance, LO leakage and (noise=True)
        clock jitter and LO phase noise applied, in that order.  ``fs`` is
        the sample rate the LO phase-noise PSD is synthesized on; it is
        required when ``lo_pn`` is set and ``noise`` is on."""
        c = self.cfg
        i = self._amp(i_code, self.amp_table_i)
        q = self._amp(q_code, self.amp_table_q)
        k = 10.0 ** (c.iq_gain_db / 20.0) * np.exp(1j * np.deg2rad(c.iq_phase_deg))
        y = i + 1j * k * q
        if c.lo_leakage_dbc is not None:
            y = y + 10.0 ** (c.lo_leakage_dbc / 20.0)
        if noise:
            rng = np.random.default_rng(seed)
            phi = np.zeros(y.shape, float)
            if c.jitter_rms_s > 0.0:
                phi += rng.normal(0.0, TWOPI * c.fout * c.jitter_rms_s, y.shape)
            if c.lo_pn is not None:
                if fs is None:
                    raise ValueError("RFDAC with lo_pn needs fs= to synthesize "
                                     "the LO phase noise")
                src = c.lo_pn.leeson("lo")
                # locked-LO approximation, as in DTCPhaseModulator: inside
                # the PLL loop BW the oscillator's f^-2/f^-3 slopes flatten
                phi += synth_from_psd(
                    lambda f: src.psd(np.maximum(f, c.lo_loop_bw)),
                    fs, y.size, rng)
            y = y * np.exp(1j * phi)
        return y

    # --------------------------------------------------------- diagnostics
    def mismatch(self) -> dict:
        """INL/DNL of each array against its own endpoint fit — the same
        ``inl_dnl`` the DPA reports, on the same raw tables."""
        return {"i": inl_dnl(self._raw_i), "q": inl_dnl(self._raw_q)}

    # --------------------------------------------------------- efficiency
    def efficiency(self, i: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Instantaneous drain efficiency for normalized axis values."""
        i, q = np.asarray(i, float), np.asarray(q, float)
        p_out = i * i + q * q
        drawn = np.abs(i) + np.abs(q)
        eta_peak = self.cfg.eff[1]
        return eta_peak * np.divide(p_out, drawn, out=np.zeros_like(p_out),
                                    where=drawn > 0)

    def average_efficiency(self, i_code: np.ndarray, q_code: np.ndarray) -> dict:
        """Modulated average ``sum(P_out) / sum(P_dc)`` over the code
        streams: ``P_out ∝ I² + Q²``, ``P_dc = (|I| + |Q|) / η_peak``.  Same
        keys as ``DPA.average_efficiency``; ``backoff_db`` is relative to an
        on-axis full-scale output (P = 1)."""
        i = self._amp(i_code, self.amp_table_i)
        q = self._amp(q_code, self.amp_table_q)
        p_out = i * i + q * q
        p_dc = (np.abs(i) + np.abs(q)) / self.cfg.eff[1]
        tot_dc = float(p_dc.sum())
        return {"eta_avg": float(p_out.sum() / tot_dc) if tot_dc else 0.0,
                "p_out_norm": float(p_out.mean()),
                "backoff_db": float(-10 * np.log10(max(p_out.mean(), 1e-30)))}
