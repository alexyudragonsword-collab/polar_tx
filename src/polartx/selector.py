"""Architecture selector: four transmitter topologies scored against one requirement.

    from polartx.selector import Requirement, select
    rep = select(Requirement(standard="WiFi7-320", bw_hz=320e6,
                             modulation="ofdm", evm_db_max=-38))
    print(rep.table())
    print(rep.recommendation)

For each candidate a technology-class template is scored *analytically*
against the requirement: a first-order EVM budget built from the same
closed-form models the chain tests check against
(``polartx.analysis.responses``, ``polartx.rfdac.iq_image_rejection_db``)
plus an integrated phase-noise term, and an average drain efficiency from the
library's own efficiency laws integrated over the modulation's amplitude
distribution.  The four candidates:

* **Narrowband ADPLL two-point** (polar) — the loop *cleans* the oscillator
  noise inside its bandwidth, so the integrated in-band phase noise is far
  lower for a narrow signal; the residual EVM floor is set by two-point
  gain/bandwidth mismatch.  But the direct-modulation DAC must cover the
  peak instantaneous frequency and stay gain-matched across it, which is
  impractical much beyond a few tens of MHz — so wideband OFDM is flagged
  infeasible (the project's central narrowband-vs-wideband split).

* **Wideband open-loop DTC** (polar) — EVM floor = DTC phase-quantization
  noise power-summed with the LO phase noise integrated over the signal
  band, plus DTC jitter and residual INL.  Feasible at any bandwidth.

* **Outphasing (LINC / Chireix)** — two constant-envelope DTC phase paths
  summed in a combiner.  It inherits the DTC's floors, but each branch runs
  at full scale regardless of the envelope, so every *independent* branch
  noise is worth ``10·log10(PAPR/2)`` dB more EVM than on the polar path
  (the shared LO's noise is common to both branches and costs the same as
  on polar, which is what ``OutphasingTX`` simulates;
  ``outphasing_shared_lo=False`` scores a separate LO per branch).  Its
  own knob is the static branch phase mismatch, whose error
  vector ``(e^{jδ} − 1)·s2`` is mostly wideband; the in-band share the OFDM
  EVM sees is a measured constant (``quad_inband_db``).  Bandwidth-agnostic
  like the DTC; the combiner sets its efficiency law.

* **Cartesian RF-DAC** — no phase path at all: two current-cell arrays on
  I/Q codes.  Floors: the shared LO phase noise, the two-axis quantization
  noise of ``dac_bits`` (spread over the ``osr`` update rate, so only
  ``1/osr`` of it lands in band), LO clock jitter, and the I/Q image at
  ``−IRR(iq_gain_err_db, iq_phase_err_deg)``.  Feasible at any bandwidth.
  Its price is the ``|I| + |Q|`` efficiency law.

**Ranking rule** (decided for stage 3 and recorded in ``cairn/LOG.md``):
infeasible candidates last; among the feasible ones, those that MEET the
EVM target rank ahead of those that miss it; inside the meeting group the
HIGHEST average efficiency wins (EVM margin beyond the target buys
nothing, efficiency does), ties broken by lower EVM; inside the missing
group the lowest EVM wins.  ``eta_avg_min`` turns efficiency into a hard
gate as well: a candidate below it is excluded with the reason in its
notes, like the ADPLL past its coverage ceiling.

The crossover is physical: at small bandwidth the loop-cleaned ADPLL wins on
integrated phase noise; at large bandwidth it runs out of two-point coverage
and the bandwidth-agnostic candidates compete on floors and efficiency.

Scores are analytic.  ``rep.best`` also names the closest ready-to-run preset
(``suggest_preset``); build it and run the real chain to confirm the shortlist
before committing.  Not scored, on purpose: DPA / RF-DAC unit-cell mismatch
(the same model on both, and calibrated out in practice), AM/PM path skew
(a calibration residual, not a technology floor), and the RF-DAC's update
images and sinc roll-off (far-out spectrum, see ``docs/architecture.md`` §7).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis.responses import dtc_quant_phase_rms, evm_db_from_phase_rms
from .rfdac import iq_image_rejection_db

TWOPI = 2.0 * np.pi
# numpy 2.0 renamed trapz -> trapezoid and dropped the old name.  Both sides
# go through getattr: a static `np.trapz` is an attribute error under numpy 2
# even though `or` never evaluates it there.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

# LC-oscillator technology class, matched to pllsim.selector: -122 dBc/Hz at
# 1 MHz offset on a 4.8 GHz carrier, scaled 20 log10(fout / 4.8 GHz).  The DCO
# used by the ADPLL direct path is ~6 dB noisier (digital tuning).
_PN_1M_REF = -122.0
_PN_REF_FOUT = 4.8e9

#: PAPR the library's chains CFR their OFDM to (``wifi_dtc`` default).
_OFDM_PAPR_DB = 8.5
#: Measured PAPR of the RRC-shaped 8DPSK burst ``waveforms.edr_dpsk`` makes
#: (3.2-3.5 dB over seeds; pinned by test_selector).
_DPSK_PAPR_DB = 3.4

ARCH_LABELS = {
    "adpll_two_point": "narrowband ADPLL two-point",
    "dtc_open_loop": "wideband open-loop DTC",
    "outphasing": "outphasing (two DTC branches, Chireix)",
    "rfdac_cartesian": "Cartesian RF-DAC",
}


def _pn_1m(fout: float, extra_db: float = 0.0) -> float:
    return _PN_1M_REF + 20.0 * np.log10(fout / _PN_REF_FOUT) + extra_db


def _leeson_sphi(f: np.ndarray, pn_1m_db: float, floor_db: float,
                 f1f3: float = 3e5) -> np.ndarray:
    """Single-sideband phase-noise PSD Sφ(f) [rad^2/Hz] of a Leeson profile:
    a 1/f^3 flicker-FM knee below ``f1f3``, a 1/f^2 white-FM region anchored at
    ``pn_1m_db`` (1 MHz), and a white-PM ``floor_db``."""
    f = np.asarray(f, dtype=float)
    s_1m = 10.0 ** (pn_1m_db / 10.0)          # 1/f^2 value at 1 MHz
    s_f2 = s_1m * (1e6 / f) ** 2              # white-FM region
    s_f3 = s_1m * (1e6 / f1f3) ** 2 * (f1f3 / f) ** 3  # flicker knee
    s = np.where(f < f1f3, s_f3, s_f2)
    return np.maximum(s, 10.0 ** (floor_db / 10.0))


def _integrated_phase_rms(pn_1m_db: float, floor_db: float,
                          f_lo: float, f_hi: float,
                          shape=None) -> float:
    """rms phase [rad] from integrating 2·Sφ(f)·|shape(f)|^2 over [f_lo, f_hi]
    (both modulation sidebands).  ``shape`` optionally applies a closed-loop
    transfer (e.g. the ADPLL high-pass DCO shaping)."""
    if f_hi <= f_lo:
        return 0.0
    f = np.logspace(np.log10(f_lo), np.log10(f_hi), 2000)
    s = _leeson_sphi(f, pn_1m_db, floor_db)
    if shape is not None:
        s = s * np.abs(shape(f)) ** 2
    var = 2.0 * _trapz(s, f)
    return float(np.sqrt(max(var, 0.0)))


@dataclass
class Requirement:
    """A transmitter requirement to rank architectures against.

    The first block describes the signal and the target; the rest are the
    technology-class assumptions each candidate is scored with — residuals
    after the calibration a product would ship with, not spec limits.
    """
    standard: str                       # label, e.g. "LTE-20" / "WiFi7-320"
    bw_hz: float                        # occupied signal bandwidth
    modulation: str = "ofdm"            # "ofdm" | "gfsk" | "dpsk" | "qam"
    evm_db_max: float = -25.0           # required EVM ceiling (dB)
    constant_envelope: bool = False     # GFSK/GMSK: envelope path is trivial
    fout: float = 6e9                   # carrier
    osr: float = 4.0                    # baseband oversampling (fs = bw·osr)
    dtc_bits: int = 11                  # DTC phase resolution to assume
    dtc_jitter_s: float = 50e-15        # DTC random edge jitter (rms)
    dtc_inl_floor_db: float = -50.0     # residual DTC INL floor after cal
    synth_loop_bw: float = 1.5e6        # synthesizer noise-optimum loop BW
    two_point_gain_match: float = 2e-3  # residual two-point gain error (0.2% cal'd)
    adpll_bw_ceiling: float = 50e6      # practical two-point coverage ceiling
    peak_slew_hz: float | None = None   # peak inst. freq; default per-modulation
    # --- shared with the two new topologies -------------------------------
    papr_db: float | None = None        # PAPR after CFR; None = per modulation
    eta_peak: float = 0.85              # PA / cell-array peak drain efficiency
    scpa_gamma: float = 0.67            # polar SCPA charging-loss parameter
    eta_avg_min: float | None = None    # required average efficiency (gate)
    # --- outphasing -------------------------------------------------------
    branch_phase_mismatch_deg: float = 1.0   # residual static branch phase error
    combiner_loss_db: float = 0.4            # combiner insertion loss
    chireix_theta_c_deg: float = 60.0        # Chireix compensation angle
    quad_inband_db: float = -7.5        # in-band share of the branch-difference
    #                                     error, measured on WiFi 160 MHz 1024-QAM
    #                                     at osr 4 (test_selector pins it)
    outphasing_shared_lo: bool = True   # both DTC branches off ONE LO (its
    #                                     phase noise is common, as OutphasingTX
    #                                     simulates); False = a separate LO per
    #                                     branch, its noise +penalty too
    # --- RF-DAC -----------------------------------------------------------
    dac_bits: int = 12                  # per-axis magnitude resolution
    dac_jitter_s: float = 50e-15        # LO clock jitter at the DAC (rms)
    iq_gain_err_db: float = 0.05        # residual I/Q gain error after cal
    iq_phase_err_deg: float = 0.5       # residual I/Q phase error after cal

    @property
    def fs_bb(self) -> float:
        """Baseband sample rate implied by the requirement, ``bw * osr``."""
        return self.bw_hz * self.osr

    @property
    def slew(self) -> float:
        """Peak instantaneous frequency the direct path must cover."""
        if self.peak_slew_hz is not None:
            return self.peak_slew_hz
        if self.constant_envelope or self.modulation in ("gfsk", "gmsk"):
            return 0.75 * self.bw_hz            # FSK peak deviation ~ 0.5·Rb
        return self.bw_hz                       # OFDM phase slews to ~±BW (P99)

    @property
    def papr(self) -> float:
        """PAPR [dB] the efficiency and the envelope-dependent floors are
        scored at: ``papr_db`` if given, else 0 for constant envelope, the
        measured 8DPSK value for "dpsk", the chains' CFR target for OFDM."""
        if self.papr_db is not None:
            return float(self.papr_db)
        if self.constant_envelope or self.modulation in ("gfsk", "gmsk"):
            return 0.0
        if self.modulation == "dpsk":
            return _DPSK_PAPR_DB
        return _OFDM_PAPR_DB


@dataclass
class Candidate:
    """One architecture scored against a requirement.

    ``terms`` is the per-contributor EVM breakdown in dB (the shared
    synthesizer term plus that architecture's own floors), which is what
    makes a verdict arguable rather than oracular: read it to see WHICH
    floor is binding before believing the ranking.  ``eta_avg`` is the
    average drain efficiency under that topology's law at the requirement's
    PAPR.  ``feasible=False`` means the architecture was excluded outright,
    with the reason in ``notes`` — a two-point ADPLL past its coverage
    ceiling, or any candidate under ``eta_avg_min``, for example.
    """

    arch: str
    evm_db: float = float("nan")
    feasible: bool = True
    eta_avg: float = float("nan")
    terms: dict = field(default_factory=dict)  # per-contributor EVM (dB)
    notes: list[str] = field(default_factory=list)

    def meets(self, evm_db_max: float) -> bool:
        return bool(self.feasible and self.evm_db <= evm_db_max)

    def sort_key(self, evm_db_max: float):
        """The ranking rule of the module docstring: feasible first, then
        meeting the target, then highest efficiency, then lowest EVM."""
        meets = self.meets(evm_db_max)
        eta = self.eta_avg if np.isfinite(self.eta_avg) else 0.0
        return (not self.feasible, not meets, -eta if meets else 0.0, self.evm_db)

    @property
    def key(self):
        """Legacy sort key (feasible first, then best EVM); the report
        ranks with ``sort_key`` against its requirement."""
        return (not self.feasible, self.evm_db)


def _combine_db(*evm_db_terms: float) -> float:
    """Power-sum EVM contributions given in dB."""
    p = sum(10.0 ** (e / 10.0) for e in evm_db_terms if np.isfinite(e))
    return float(10.0 * np.log10(p)) if p > 0 else float("-inf")


def _hp_shape(loop_bw: float):
    """First-order high-pass DCO shaping |H_hp(f)| = f/sqrt(f^2+fbw^2): the
    synthesizer loop suppresses DCO noise below its bandwidth (and, being
    high-pass, kills the near-carrier 1/f^3 flicker that would otherwise
    dominate the in-band integral)."""
    def shape(f):
        return f / np.sqrt(f ** 2 + loop_bw ** 2)
    return shape


def _synth_pn_evm(req: Requirement) -> float:
    """In-band EVM (dB) from the shared synthesizer: DCO noise loop-cleaned
    below ``synth_loop_bw`` and integrated over the signal band.  All four
    architectures use the same synthesizer (the RF-DAC's LO included), so
    this term is common — what separates them is the *extra* floors below."""
    dco_pn = _pn_1m(req.fout, extra_db=6.0)      # DCO class, ~6 dB over LC
    pn_rms = _integrated_phase_rms(dco_pn, dco_pn - 33.0,
                                   f_lo=1e3, f_hi=max(req.bw_hz / 2.0, 2e3),
                                   shape=_hp_shape(req.synth_loop_bw))
    return evm_db_from_phase_rms(pn_rms)


# ------------------------------------------------------------ efficiency
def amplitude_distribution(papr_db: float, n: int = 2001):
    """Normalized envelope ``a = |x| / A_max`` and its probability weights
    for a signal of the given PAPR: a complex-Gaussian (Rayleigh) envelope
    clipped at the peak, i.e. what CFR leaves of an OFDM burst — the mass
    the clip removes sits at ``a = 1``.  PAPR 0 is a constant envelope.
    ``sum(w) == 1``.  Above ~6 dB the distribution's own PAPR is the
    parameter to within 1 %; at the 8DPSK 3.4 dB the clipped mass is 11 %
    and its own PAPR is ~3.9 dB — a first-order screen, not a waveform."""
    if papr_db <= 0.0:
        return np.array([1.0]), np.array([1.0])
    s = 10.0 ** (-papr_db / 10.0)                # mean power / peak power
    a = np.linspace(0.0, 1.0, n)
    pdf = 2.0 * a / s * np.exp(-a * a / s)
    w = pdf * (a[1] - a[0])
    w[0] *= 0.5
    w[-1] = 0.5 * w[-1] + np.exp(-1.0 / s)        # clipped tail -> the peak
    return a, w / w.sum()


def _eta_from_pdc(a: np.ndarray, w: np.ndarray, p_dc: np.ndarray,
                  loss_db: float = 0.0) -> float:
    """Modulated average ``E[P_out] / E[P_dc]`` with ``P_out = a²`` (times an
    insertion loss) — the same convention as every ``average_efficiency``
    in the library."""
    return float(np.sum(w * a * a) * 10.0 ** (-loss_db / 10.0) / np.sum(w * p_dc))


def eta_polar_scpa(req: Requirement) -> float:
    """Polar DPA under the SCPA law ``η = η_peak·a²/(a² + γ·a·(1−a))``:
    ``P_dc = (a² + γ·a·(1−a)) / η_peak``."""
    a, w = amplitude_distribution(req.papr)
    p_dc = (a * a + req.scpa_gamma * a * (1.0 - a)) / req.eta_peak
    return _eta_from_pdc(a, w, p_dc)


def eta_outphasing_chireix(req: Requirement) -> float:
    """Two branch PAs at full scale (``η_pa = η_peak``) into a Chireix
    combiner: ``P_dc = sqrt(cos⁴θ + b_eff²) / η_pa`` with ``cos θ = a`` and
    ``b_eff = ½·sin 2θ − ½·sin 2θ_c`` — ``OutphasingCombiner``'s law."""
    a, w = amplitude_distribution(req.papr)
    th = np.arccos(np.clip(a, 0.0, 1.0))
    b = 0.5 * np.sin(2.0 * np.deg2rad(req.chireix_theta_c_deg))
    b_eff = 0.5 * np.sin(2.0 * th) - b
    p_dc = np.sqrt(a ** 4 + b_eff ** 2) / req.eta_peak
    return _eta_from_pdc(a, w, p_dc, loss_db=req.combiner_loss_db)


def eta_rfdac_iq_cells(req: Requirement) -> float:
    """RF-DAC ``iq_cells`` law averaged over a uniform carrier phase: the
    arrays draw ``|I| + |Q| = a·(|cos φ| + |sin φ|)``, whose mean is
    ``4a/π``, so ``P_dc = (4/π)·a / η_peak``."""
    a, w = amplitude_distribution(req.papr)
    p_dc = (4.0 / np.pi) * a / req.eta_peak
    return _eta_from_pdc(a, w, p_dc)


# ---------------------------------------------------------------- scoring
def _dtc_floors(req: Requirement) -> dict:
    """The open-loop DTC's own EVM floors, shared by the two topologies
    that run the modulation through a DTC phase path."""
    q_rms = dtc_quant_phase_rms(req.dtc_bits, range_ui=1.0, osr=req.osr)
    jit_rad = TWOPI * req.fout * req.dtc_jitter_s
    return {"dtc_quant": evm_db_from_phase_rms(q_rms),
            "dtc_jitter": evm_db_from_phase_rms(jit_rad),
            "dtc_inl": req.dtc_inl_floor_db}


def _score_dtc(req: Requirement, synth_evm: float) -> Candidate:
    c = Candidate("dtc_open_loop")
    # extra floors the open-loop DTC adds on top of the shared synth PN
    c.terms = {"synth_pn": synth_evm, **_dtc_floors(req)}
    c.evm_db = _combine_db(*c.terms.values())
    c.eta_avg = eta_polar_scpa(req)
    c.notes.append(f"{req.dtc_bits}-bit DTC, {req.dtc_jitter_s*1e15:.0f} fs "
                   f"jitter; bandwidth-agnostic (open loop)")
    return c


def _score_adpll(req: Requirement, synth_evm: float) -> Candidate:
    c = Candidate("adpll_two_point")
    # ADPLL imprints the modulation *inside* the loop (analog-resolution FM):
    # no DTC quantization, jitter or INL floor.  The residual is the shared
    # synth PN plus two-point gain/bandwidth mismatch, which only bites on the
    # phase-path energy the direct (high-pass) path carries above the loop BW.
    frac_hp = float(np.clip(1.0 - req.synth_loop_bw / max(req.bw_hz / 2.0,
                                                          req.synth_loop_bw),
                            0.0, 1.0))
    eps = req.two_point_gain_match                # residual two-point match
    evm_mismatch = (20.0 * np.log10(eps) + 10.0 * np.log10(frac_hp)
                    if frac_hp > 0 else float("-inf"))
    c.terms = {"synth_pn": synth_evm, "two_point_mismatch": evm_mismatch}
    c.evm_db = _combine_db(synth_evm, evm_mismatch)
    c.eta_avg = eta_polar_scpa(req)
    c.notes.append("in-loop FM: no DTC quant/jitter/INL floor")
    # feasibility: the direct-modulation DAC must cover the peak slew and stay
    # gain-matched across the phase-path bandwidth the two paths must track.
    if req.bw_hz > req.adpll_bw_ceiling:
        c.feasible = False
        c.notes.append(
            f"signal BW {req.bw_hz/1e6:.0f} MHz exceeds practical two-point "
            f"coverage (~{req.adpll_bw_ceiling/1e6:.0f} MHz): the direct FM "
            f"DAC cannot stay gain-matched over the phase-path bandwidth")
    elif req.slew > 0.6 * req.adpll_bw_ceiling:
        c.notes.append(
            f"peak slew {req.slew/1e6:.0f} MHz approaching the coverage "
            f"ceiling — direct-DAC range/linearity is the binding constraint")
    return c


def outphasing_branch_penalty_db(papr_db: float) -> float:
    """How much more EVM an *independent* branch noise costs on outphasing
    than on the polar path: both branches sit at ``A_max/2`` whatever the
    envelope, so the error power is ``φ²·A_max²/2`` against the polar
    ``φ²·E|x|² = φ²·A_max²/PAPR`` — ``10·log10(PAPR/2)`` dB.  Negative
    (−3 dB) for a constant envelope: two half-amplitude branches average
    their independent noise."""
    return float(10.0 * np.log10(10.0 ** (papr_db / 10.0) / 2.0))


def outphasing_mismatch_evm_db(delta_deg: float, papr_db: float,
                               quad_inband_db: float) -> float:
    """Closed-form EVM of a static branch phase error ``δ``: the error
    vector ``(e^{jδ} − 1)·s2`` splits into ``x/2`` (a scalar gain the EVM
    equalizer removes) and the quadrature term ``q = ½·A_max·sin θ·e^{jφ}``,
    whose power is ``¼·A_max²·(1 − 1/PAPR)``; relative to the signal that
    is ``δ²·(PAPR − 1)/4``, of which only ``quad_inband_db`` lands in band
    (the term is as wide as a constant-envelope branch).  ``−inf`` for a
    constant envelope, where ``q ≡ 0``."""
    papr = 10.0 ** (papr_db / 10.0)
    if papr <= 1.0:
        return float("-inf")
    d = np.deg2rad(delta_deg)
    return float(20.0 * np.log10(max(d, 1e-15))
                 + 10.0 * np.log10((papr - 1.0) / 4.0) + quad_inband_db)


def _score_outphasing(req: Requirement, synth_evm: float) -> Candidate:
    c = Candidate("outphasing")
    pen = outphasing_branch_penalty_db(req.papr)
    floors = {k + "+branches": v + pen for k, v in _dtc_floors(req).items()}
    mism = outphasing_mismatch_evm_db(req.branch_phase_mismatch_deg, req.papr,
                                      req.quad_inband_db)
    if req.outphasing_shared_lo:
        synth = {"synth_pn": synth_evm}           # common to both branches
    else:
        synth = {"synth_pn+branches": synth_evm + pen}
    c.terms = {**synth, **floors, "branch_mismatch": mism}
    c.evm_db = _combine_db(*c.terms.values())
    c.eta_avg = eta_outphasing_chireix(req)
    c.notes.append(
        f"two full-scale DTC branches: independent floors {pen:+.1f} dB at "
        f"{req.papr:.1f} dB PAPR; {req.branch_phase_mismatch_deg:.1f} deg "
        f"branch mismatch; Chireix {req.chireix_theta_c_deg:.0f} deg, "
        f"{req.combiner_loss_db:.1f} dB combiner")
    return c


def rfdac_quant_evm_db(dac_bits: int, papr_db: float, osr: float) -> float:
    """Two-axis quantization noise of a sign-magnitude I/Q DAC with the
    on-axis full scale at the signal peak: noise ``2·Δ²/12`` with
    ``Δ = 1/(2^n − 1)``, of which ``1/osr`` lands in the signal band,
    against a signal power ``1/PAPR``."""
    delta = 1.0 / ((1 << dac_bits) - 1)
    noise_inband = 2.0 * delta * delta / 12.0 / osr
    return float(10.0 * np.log10(noise_inband * 10.0 ** (papr_db / 10.0)))


def _score_rfdac(req: Requirement, synth_evm: float) -> Candidate:
    c = Candidate("rfdac_cartesian")
    evm_q = rfdac_quant_evm_db(req.dac_bits, req.papr, req.osr)
    evm_jit = evm_db_from_phase_rms(TWOPI * req.fout * req.dac_jitter_s)
    evm_img = -iq_image_rejection_db(req.iq_gain_err_db, req.iq_phase_err_deg)
    c.terms = {"synth_pn": synth_evm, "dac_quant": evm_q,
               "dac_jitter": evm_jit, "iq_image": evm_img}
    c.evm_db = _combine_db(*c.terms.values())
    c.eta_avg = eta_rfdac_iq_cells(req)
    c.notes.append(
        f"{req.dac_bits}-bit I/Q arrays at {req.fs_bb/1e6:.0f} MS/s, "
        f"{req.dac_jitter_s*1e15:.0f} fs LO jitter, I/Q residual "
        f"{req.iq_gain_err_db:.2f} dB / {req.iq_phase_err_deg:.1f} deg "
        f"(IRR {-evm_img:.0f} dB); no phase path, bandwidth-agnostic")
    return c


def _apply_efficiency_gate(req: Requirement, c: Candidate) -> None:
    if req.eta_avg_min is not None and c.feasible and c.eta_avg < req.eta_avg_min:
        c.feasible = False
        c.notes.append(f"average efficiency {c.eta_avg*100:.0f} % under the "
                       f"required {req.eta_avg_min*100:.0f} %")


@dataclass
class SelectorReport:
    """The scored comparison, plus renderings of it.

    ``best`` / ``recommendation`` give the verdict, ``table()`` gives the
    breakdown, ``suggest_preset()`` names the closest ready-to-run preset
    so a recommendation can be run rather than just read.

    This is an ANALYTIC screen on a shared-synthesizer basis, not a
    simulation: it says which architecture to reach for first.  Confirm
    with an actual chain run before committing to it.
    """

    req: Requirement
    candidates: list[Candidate]

    def ranked(self) -> list[Candidate]:
        """Candidates in verdict order (the module's ranking rule)."""
        return sorted(self.candidates,
                      key=lambda c: c.sort_key(self.req.evm_db_max))

    @property
    def best(self) -> Candidate | None:
        """Top-ranked feasible candidate, or None if none is feasible."""
        top = self.ranked()[0] if self.candidates else None
        return top if top is not None and top.feasible else None

    @property
    def recommendation(self) -> str:
        """One-line verdict: architecture, EVM, margin vs target, efficiency,
        and why anything was excluded."""
        b = self.best
        if b is None:
            return "no feasible architecture for this requirement"
        meets = b.evm_db <= self.req.evm_db_max
        margin = self.req.evm_db_max - b.evm_db
        verb = "meets" if meets else "MISSES"
        arch = ARCH_LABELS.get(b.arch, b.arch)
        excl = [c for c in self.candidates if not c.feasible]
        why = ""
        if excl:
            why = ("  ("
                   + "; ".join(f"{c.arch} excluded: {c.notes[-1]}"
                               for c in excl) + ")")
        return (f"recommend {arch}: EVM ~{b.evm_db:.1f} dB {verb} the "
                f"{self.req.evm_db_max:.0f} dB target "
                f"(margin {margin:+.1f} dB), average efficiency "
                f"~{b.eta_avg*100:.0f} %.{why}")

    def suggest_preset(self) -> str:
        """Name of the closest ready-to-run preset for the best architecture."""
        b = self.best
        if b is None:
            return ""
        if b.arch == "adpll_two_point":
            if self.req.constant_envelope or self.req.modulation in ("gfsk",):
                rate = 2e6 if self.req.bw_hz > 1.5e6 else 1e6
                return f"ble_adpll(rate={rate:.0g})"
            return "lte20_adpll(...)"
        if b.arch == "outphasing":
            return f"wifi_outphasing(bw={self.req.bw_hz:.0g})"
        if b.arch == "rfdac_cartesian":
            return f"wifi_rfdac(bw={self.req.bw_hz:.0g}, n_bits={self.req.dac_bits})"
        std = self.req.standard.lower()
        if "nr" in std or "5g" in std:
            return f"nr_dtc(bw={self.req.bw_hz:.0g})"
        return f"wifi_dtc(bw={self.req.bw_hz:.0g}, n_bits={self.req.dtc_bits})"

    def table(self) -> str:
        """Fixed-width comparison table in verdict order: EVM, PASS/fail vs
        target, average efficiency, and the per-contributor breakdown."""
        w = 18
        lines = [f"{'arch':{w}s}{'EVM':>9s}{'target':>9s}{'eta':>7s}  "
                 "breakdown / notes"]
        for c in self.ranked():
            eta = f"{c.eta_avg*100:.0f}%" if np.isfinite(c.eta_avg) else "-"
            if c.feasible:
                mark = "PASS" if c.evm_db <= self.req.evm_db_max else "fail"
                terms = ", ".join(f"{k} {v:.0f}" for k, v in c.terms.items())
                lines.append(f"{c.arch:{w}s}{c.evm_db:7.1f}dB{mark:>9s}{eta:>7s}  "
                             f"[{terms}] " + "; ".join(c.notes))
            else:
                lines.append(f"{c.arch:{w}s}{'-':>9s}{'excl.':>9s}{eta:>7s}  "
                             + "; ".join(c.notes))
        return "\n".join(lines)


def select(req: Requirement) -> SelectorReport:
    """Rank the four transmitter topologies against ``req`` (analytic scoring)."""
    synth_evm = _synth_pn_evm(req)
    cands = [_score_adpll(req, synth_evm), _score_dtc(req, synth_evm),
             _score_outphasing(req, synth_evm), _score_rfdac(req, synth_evm)]
    for c in cands:
        _apply_efficiency_gate(req, c)
    return SelectorReport(req=req, candidates=cands)
