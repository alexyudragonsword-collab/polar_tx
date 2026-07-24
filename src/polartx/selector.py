"""Architecture selector: narrowband ADPLL two-point vs wideband open-loop DTC.

    from polartx.selector import Requirement, select
    rep = select(Requirement(standard="WiFi7-320", bw_hz=320e6,
                             modulation="ofdm", evm_db_max=-38))
    print(rep.table())
    print(rep.recommendation)

For each of the two polar phase-path architectures a technology-class template
is scored *analytically* against the requirement and the winner recommended.
The scoring is a first-order EVM budget built from the same closed-form models
the chain tests check against (``polartx.analysis.responses``) plus an
integrated phase-noise term:

* **Wideband open-loop DTC** — EVM floor = DTC phase-quantization noise
  (``dtc_quant_phase_rms``) power-summed with the LO phase noise integrated
  over the signal band.  The LO is PLL-locked but the modulation is applied
  open-loop, so above the LO's loop bandwidth the full oscillator noise lands
  in-band: wide signals integrate more of it.  Feasible at any bandwidth.

* **Narrowband ADPLL two-point** — the loop *cleans* the oscillator noise
  inside its bandwidth, so the integrated in-band phase noise is far lower for
  a narrow signal; the residual EVM floor is set by two-point gain/bandwidth
  mismatch.  But the direct-modulation DAC must cover the peak instantaneous
  frequency and stay gain-matched across it, which is impractical much beyond
  a few tens of MHz — so wideband OFDM is flagged infeasible (the project's
  central narrowband-vs-wideband split).

The crossover is physical: at small bandwidth the loop-cleaned ADPLL wins on
integrated phase noise; at large bandwidth it runs out of two-point coverage
and the open-loop DTC — quantization-limited but bandwidth-agnostic — is the
only choice.

Scores are analytic.  ``rep.best`` also names the closest ready-to-run preset
(``suggest_preset``); build it and run the real chain to confirm the shortlist
before committing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis.responses import dtc_quant_phase_rms, evm_db_from_phase_rms

TWOPI = 2.0 * np.pi
_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x renamed it

# LC-oscillator technology class, matched to pllsim.selector: -122 dBc/Hz at
# 1 MHz offset on a 4.8 GHz carrier, scaled 20 log10(fout / 4.8 GHz).  The DCO
# used by the ADPLL direct path is ~6 dB noisier (digital tuning).
_PN_1M_REF = -122.0
_PN_REF_FOUT = 4.8e9


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
    """A transmitter requirement to rank architectures against."""
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

    @property
    def fs_bb(self) -> float:
        return self.bw_hz * self.osr

    @property
    def slew(self) -> float:
        """Peak instantaneous frequency the direct path must cover."""
        if self.peak_slew_hz is not None:
            return self.peak_slew_hz
        if self.constant_envelope or self.modulation in ("gfsk", "gmsk"):
            return 0.75 * self.bw_hz            # FSK peak deviation ~ 0.5·Rb
        return self.bw_hz                       # OFDM phase slews to ~±BW (P99)


@dataclass
class Candidate:
    arch: str
    evm_db: float = float("nan")
    feasible: bool = True
    terms: dict = field(default_factory=dict)  # per-contributor EVM (dB)
    notes: list[str] = field(default_factory=list)

    @property
    def key(self):
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
    below ``synth_loop_bw`` and integrated over the signal band.  Both
    architectures use the same synthesizer, so this term is common — what
    separates them is the *extra* floors below."""
    dco_pn = _pn_1m(req.fout, extra_db=6.0)      # DCO class, ~6 dB over LC
    pn_rms = _integrated_phase_rms(dco_pn, dco_pn - 33.0,
                                   f_lo=1e3, f_hi=max(req.bw_hz / 2.0, 2e3),
                                   shape=_hp_shape(req.synth_loop_bw))
    return evm_db_from_phase_rms(pn_rms)


def _score_dtc(req: Requirement, synth_evm: float) -> Candidate:
    c = Candidate("dtc_open_loop")
    # extra floors the open-loop DTC adds on top of the shared synth PN:
    q_rms = dtc_quant_phase_rms(req.dtc_bits, range_ui=1.0, osr=req.osr)
    evm_q = evm_db_from_phase_rms(q_rms)                       # quantization
    jit_rad = TWOPI * req.fout * req.dtc_jitter_s              # edge jitter
    evm_jit = evm_db_from_phase_rms(jit_rad)
    evm_inl = req.dtc_inl_floor_db                             # residual INL
    c.terms = {"synth_pn": synth_evm, "dtc_quant": evm_q,
               "dtc_jitter": evm_jit, "dtc_inl": evm_inl}
    c.evm_db = _combine_db(synth_evm, evm_q, evm_jit, evm_inl)
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


@dataclass
class SelectorReport:
    req: Requirement
    candidates: list[Candidate]

    @property
    def best(self) -> Candidate | None:
        ok = [c for c in self.candidates if c.feasible]
        return min(ok, key=lambda c: c.evm_db) if ok else None

    @property
    def recommendation(self) -> str:
        b = self.best
        if b is None:
            return "no feasible architecture for this requirement"
        meets = b.evm_db <= self.req.evm_db_max
        margin = self.req.evm_db_max - b.evm_db
        verb = "meets" if meets else "MISSES"
        arch = "narrowband ADPLL two-point" if b.arch == "adpll_two_point" \
            else "wideband open-loop DTC"
        excl = [c for c in self.candidates if not c.feasible]
        why = ""
        if excl:
            why = ("  ("
                   + "; ".join(f"{c.arch} excluded: {c.notes[-1]}"
                               for c in excl) + ")")
        return (f"recommend {arch}: EVM ~{b.evm_db:.1f} dB {verb} the "
                f"{self.req.evm_db_max:.0f} dB target "
                f"(margin {margin:+.1f} dB).{why}")

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
        std = self.req.standard.lower()
        if "nr" in std or "5g" in std:
            return f"nr_dtc(bw={self.req.bw_hz:.0g})"
        return f"wifi_dtc(bw={self.req.bw_hz:.0g}, n_bits={self.req.dtc_bits})"

    def table(self) -> str:
        w = 18
        lines = [f"{'arch':{w}s}{'EVM':>9s}{'target':>9s}  breakdown / notes"]
        for c in sorted(self.candidates, key=lambda c: c.key):
            if c.feasible:
                mark = "PASS" if c.evm_db <= self.req.evm_db_max else "fail"
                terms = ", ".join(f"{k} {v:.0f}" for k, v in c.terms.items())
                lines.append(f"{c.arch:{w}s}{c.evm_db:7.1f}dB{mark:>9s}  "
                             f"[{terms}] " + "; ".join(c.notes))
            else:
                lines.append(f"{c.arch:{w}s}{'-':>9s}{'excl.':>9s}  "
                             + "; ".join(c.notes))
        return "\n".join(lines)


def select(req: Requirement) -> SelectorReport:
    """Rank the two polar architectures against ``req`` (analytic scoring)."""
    synth_evm = _synth_pn_evm(req)
    cands = [_score_adpll(req, synth_evm), _score_dtc(req, synth_evm)]
    return SelectorReport(req=req, candidates=cands)
