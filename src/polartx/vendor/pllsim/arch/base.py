# Vendored from pll_simulator@931cfaf: src/pllsim/arch/base.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Common architecture contract."""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from ..core.jitter import ldbc_from_sphi
from ..core.results import AnalysisResult, SimResult


class PLLBase(ABC):
    """analyze() = linear phase-domain model; simulate() = behavioral time domain.

    Every engine is constructed from a config dataclass and keeps it on `cfg`,
    and the whole library relies on that -- corners, Monte Carlo, the GUIs and
    the selector all reach for `pll.cfg`.  Declaring it here makes the contract
    part of the base class instead of a convention six subclasses happen to
    share; the type is deliberately loose because each engine has its own
    config type and they have no common base.
    """

    cfg: Any

    @abstractmethod
    def analyze(self, f: np.ndarray | None = None) -> AnalysisResult: ...

    @abstractmethod
    def simulate(self, n_cycles: int, *, noise: bool = True,
                 calibration: bool = True, seed: int = 0) -> SimResult: ...

    def design_report(self, ar: AnalysisResult | None = None) -> str:
        ar = ar or self.analyze()
        lines = [
            f"=== {type(self).__name__} design report ===",
            f"fout = {ar.f0 / 1e9:.6f} GHz",
            f"UGB = {ar.loop.f_ugb / 1e6:.3f} MHz   PM = {ar.loop.pm_deg:.1f} deg   "
            f"GM = {ar.loop.gm_db:.1f} dB",
            f"closed-loop f-3dB = {ar.loop.f_3db / 1e6:.3f} MHz   "
            f"peaking = {ar.loop.peaking_db:.2f} dB",
            f"RMS jitter ({ar.int_band[0]:.0f} Hz - {ar.int_band[1] / 1e6:.0f} MHz) "
            f"= {ar.jitter_fs:.1f} fs   IPN = {ar.ipn_dbc:.1f} dBc",
            f"dominant noise source: {ar.dominant_source()}",
        ]
        i = np.searchsorted(ar.f, 1e6)
        if i < ar.f.size:
            lines.append(f"L(1 MHz) total = {ldbc_from_sphi(ar.pn_breakdown['total'][i]):.1f} dBc/Hz")
        for k, v in ar.spurs_analytic.items():
            lines.append(f"spur[{k}] = {v:.1f} dBc (analytic)")
        for note in ar.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)


def start_offset_kwarg(pll) -> str | None:
    """The simulate() keyword that starts this architecture's oscillator off-target.

    ILCM and MDLL call it ``f_free_error``: their oscillator runs free and the
    FTL corrects it, so nothing "starts" off-target the way a locked loop does.
    The other architectures call the same quantity ``f_start_offset``.  Anything
    that hops a channel or sweeps a starting error has to ask, rather than
    assume one name and fail at the call.  None means the engine has no such
    knob at all.
    """
    params = inspect.signature(type(pll).simulate).parameters
    for name in ("f_start_offset", "f_free_error"):
        if name in params:
            return name
    return None


def supply_ripple_v(ripple, n_cycles: int, tref: float):
    """Per-cycle supply deviation [V] from a (amplitude_v, freq_hz) sine.

    Shared so that every engine spells supply pushing the same way: the knob
    lives on the common OscConfig (pushing_hz_v), and for a long time only the
    CPPLL actually read it, which made it a decoration on five architectures.
    """
    if ripple is None:
        return np.zeros(n_cycles)
    amp, f_sup = ripple
    return amp * np.sin(2.0 * np.pi * f_sup * np.arange(n_cycles) * tref)


def pull_hz(osc_cfg, n_cycles: int, tref: float) -> np.ndarray:
    """Per-cycle oscillator frequency perturbation from an aggressor [Hz].

    Shared for the same reason supply_ripple_v is: the knob lives on the common
    OscConfig, so every engine has to read it or it is a decoration.  Pulling
    is coupling into the tank, not into the loop, so it applies to all six
    architectures -- including the injection-locked ones, where the same
    realignment that highpasses oscillator noise also suppresses this.
    """
    if not getattr(osc_cfg, "pulled", False):
        return np.zeros(n_cycles)
    t = np.arange(n_cycles) * tref
    return osc_cfg.pull_lock_range_hz * np.sin(
        2.0 * np.pi * osc_cfg.pull_offset_hz * t)


def add_pull_offset(offsets, osc_cfg, fref: float):
    """Add the pulling beat to a spur-offset list when it is observable.

    Above fref/2 a reference-rate record cannot resolve it -- that offset is
    left out rather than reported at its alias, which would be a wrong number
    rather than a missing one.
    """
    if getattr(osc_cfg, "pulled", False) and osc_cfg.pull_offset_hz < 0.45 * fref:
        return (list(offsets) if offsets else []) + [osc_cfg.pull_offset_hz]
    return offsets


def pull_spur(osc_cfg, err_fr=None) -> dict[str, float]:
    """{"pull_spur": dBc} for an aggressor, after the loop's rejection.

    Empty when nothing is configured -- an absent aggressor is an absent key,
    not a number.  Also empty (with the caller warned separately) when the
    aggressor is inside the lock range, where the oscillator is captured and
    the weak-pulling expression does not describe anything.
    """
    if not getattr(osc_cfg, "pulled", False) or osc_cfg.within_lock_range():
        return {}
    gain = 1.0
    if err_fr is not None:
        gain = float(np.interp(osc_cfg.pull_offset_hz, err_fr.f,
                               np.abs(err_fr.h)))
    return {"pull_spur": osc_cfg.pull_spur_dbc(gain)}


def pull_notes(osc_cfg) -> list[str]:
    if not getattr(osc_cfg, "pulled", False):
        return []
    if osc_cfg.within_lock_range():
        return [f"aggressor is {osc_cfg.pull_offset_hz / 1e6:.2f} MHz away but "
                f"the lock range is {osc_cfg.pull_lock_range_hz / 1e6:.2f} MHz: "
                "the oscillator is CAPTURED, not pulled — no sideband is "
                "reported because the loop no longer owns the frequency"]
    return [f"injection pulling: f_L={osc_cfg.pull_lock_range_hz / 1e6:.2f} MHz "
            f"aggressor at {osc_cfg.pull_offset_hz / 1e6:.2f} MHz offset "
            "(coupling into the tank — the loop filter cannot fix it)"]


def attach_fine(sim: SimResult, fine: np.ndarray, m_os: int, fref: float,
                int_band: tuple[float, float], offsets=None) -> SimResult:
    """Re-derive the spur table and jitter from an oversampled phase record.

    A record sampled once per reference edge cannot show anything at fref: the
    reference spur sits exactly at that record's sampling rate, so it aliases
    to DC and reads as spurless.  Every engine that wants its reference spur to
    be observable has to keep an intra-period phase trace, and every one of
    them then post-processes it identically — hence this.

    ``jitter_fs`` is replaced, because the oversampled record is the honest one:
    it contains the intra-period ripple that the reference-rate record drops.
    """
    from ..core.jitter import rms_jitter_fs
    from ..core.spectrum import find_spurs, periodogram_psd, phase_psd
    fs_fine = m_os * fref
    n0 = fine.size // 4
    f_p, s_p = periodogram_psd(fine[n0:], fs_fine)
    sim.extra["fine_fs"] = fs_fine
    sim.extra["fine_f"], sim.extra["fine_psd"] = f_p, s_p
    # Two estimators of the same record, each for what it is good at.  The
    # full-length periodogram above resolves the spurs; it is also a single
    # realization, ~5.6 dB of spread per bin, which is far too coarse to
    # compare against a model band by band -- one ILCM point read +2.37,
    # +1.56, +1.22, -0.50 and +1.97 dB in its lowest band across five seeds.
    # The Welch estimate below is what the reference-rate record already used
    # and what the comparison tolerances were set against.  Same record, so
    # the picture and the number still describe one run; different estimator,
    # because a spur table and a band average want opposite things.
    f_w, s_w = phase_psd(fine[n0:], fs_fine)
    sim.extra["fine_f_avg"], sim.extra["fine_psd_avg"] = f_w, s_w
    want = [fref, 2.0 * fref] + list(offsets or [])
    sim.spurs_fft.update(find_spurs(f_p, s_p, [o for o in want
                                               if 0 < o < 0.45 * fs_fine]))
    sim.jitter_fs = rms_jitter_fs(f_p, s_p, sim.f0,
                                  max(int_band[0], f_p[0]),
                                  min(int_band[1], 0.45 * fs_fine))
    sim.notes.append(f"jitter integrated on the {m_os}x oversampled phase "
                     "(includes intra-period ripple)")
    return sim


def dtc_t_target_of(pll) -> Callable[[float], float]:
    """The DTC target each fractional architecture asks for, per MASH residue.

    One mapping per architecture, in one place: the SSPLL delays the
    reference edge to the *next* VCO edge, less half the DTC range so the
    codes sit mid-scale; the SPLL delays the divided edge by minus the
    residue on the same bipolar offset; the CPPLL and ADPLL delay by the
    residue itself.  The engines' per-cycle loops keep the same expression
    inline for speed; analyze() and every GUI page get it from here, which is
    what stops the five copies this used to be from disagreeing.
    """
    c = pll.cfg
    if getattr(c, "frac", None) is None or c.frac.dtc is None:
        raise TypeError(f"{type(pll).__name__} has no DTC to map a residue onto")
    fout, half = float(c.fout), c.frac.dtc.range_s / 2.0
    kind = type(pll).__name__
    if kind == "SSPLL":
        return lambda r: (1.0 + r) / fout - half
    if kind == "SPLL":
        return lambda r: -r / fout - half
    return lambda r: r / fout


def tuning_notes(osc_cfg, v_needed: float, what: str = "fout") -> list[str]:
    """What the control-voltage law says about reaching a frequency.

    Two notes, exclusive: the range is set and the target lies outside it
    (the varactor rails, the loop cannot get there), or no range is set and
    the target needs more travel than any single band supplies (the
    unbounded law follows it, a real oscillator would not).  Shared by the
    three analog architectures so the wording cannot drift between them.
    """
    from ..core.boundaries import (
        TUNING_SWING_V,
        tuning_law_railed,
        tuning_swing_exceeded,
    )
    lo, hi = osc_cfg.v_min, osc_cfg.v_max
    if tuning_law_railed(v_needed, lo, hi):
        rail = f"v_min={lo:g} V" if lo is not None and v_needed < lo else f"v_max={hi:g} V"
        return [f"{what} needs v_ctrl={v_needed:+.2f} V but the law is "
                f"limited at {rail}: the varactor rails and the loop cannot "
                f"reach {what} -- the linear model is evaluated at a point "
                "the loop cannot occupy, and a run reads never-locked"]
    if tuning_swing_exceeded(v_needed, lo, hi):
        return [f"tuning law unbounded (no v_min/v_max): {what} needs "
                f"v_ctrl={v_needed:+.2f} V from f0, beyond the "
                f"+/-{TUNING_SWING_V:g} V no single varactor band spans -- "
                "the loop follows it here because nothing stops it; set "
                "v_min/v_max, or a coarse band bank (n_bands/band_step_hz), "
                "to model the oscillator you can build"]
    return []


def tuning_sim_notes(sim: SimResult, osc_cfg) -> SimResult:
    """The same statement from a run's own control-voltage record.

    `postprocess` already says "never reached fout" when a railed loop parks
    off-frequency; this adds *why* (which rail), and for an unbounded law it
    says when the loop parked beyond any credible swing -- which `postprocess`
    cannot see, because the unbounded loop does reach fout.
    """
    import numpy as np

    from ..core.boundaries import LOCK_TAIL_CYCLES, TUNING_SWING_V
    ctrl = getattr(sim, "ctrl", None)
    if ctrl is None or len(ctrl) == 0:
        return sim
    n_tail = max(1, min(LOCK_TAIL_CYCLES, len(ctrl) // 4))
    v = float(np.mean(np.asarray(ctrl[-n_tail:], dtype=float)))
    lo, hi = osc_cfg.v_min, osc_cfg.v_max
    if lo is not None and v <= lo + 1e-9:
        sim.notes.append(f"control voltage railed at v_min={lo:g} V over the "
                         "tail of the run: the target is below the tuning range")
    elif hi is not None and v >= hi - 1e-9:
        sim.notes.append(f"control voltage railed at v_max={hi:g} V over the "
                         "tail of the run: the target is above the tuning range")
    elif lo is None and hi is None and abs(v) > TUNING_SWING_V:
        sim.notes.append(f"tuning law unbounded (no v_min/v_max): the loop "
                         f"parked at v_ctrl={v:+.2f} V, beyond the "
                         f"+/-{TUNING_SWING_V:g} V no single varactor band "
                         "spans -- a real oscillator would have railed")
    return sim


def flicker_corner_hz(c) -> float:
    """Highest configured 1/f corner: reference, divider (if any), oscillator.

    Zero when the configuration carries no flicker at all, which is what
    lets postprocess keep its synthesis-floor note quiet for white-only runs.
    """
    return max(float(getattr(c, "ref_pn_fc", 0.0) or 0.0),
               float(getattr(c, "div_pn_fc", 0.0) or 0.0),   # ADPLL: None = no divider
               float(getattr(c.osc, "pn_f1f3", 0.0)))


def no_fine_note(sim: SimResult) -> SimResult:
    """Say plainly that the reference spur is not in this record."""
    sim.notes.append("jitter integrated at the reference rate: intra-period "
                     "ripple is NOT included and the reference spur aliases "
                     "to DC — pass fine_oversample>1 to capture both")
    return sim


def run_band_select(osc, cfg, rng, noise: bool, enabled: bool = True):
    """Binary coarse-band search before the loop closes.

    Returns the trace, or None when the bank is a single band or the caller
    disabled it.  Lives here because the search is identical for every
    architecture whose oscillator has a control voltage — it was CPPLL-only
    for a while, which made n_bands a decoration on the other analog loops
    while `export` happily emitted a band-search FSM for them.
    """
    if cfg.osc.n_bands <= 1 or not enabled:
        return None
    from ..calibration.gain_cal import BandSelect
    bs = BandSelect(cfg.osc.n_bands, cfg.fout)
    while not bs.done:
        osc.band = bs.band
        f_meas = osc.freq(0.0)
        if noise:      # counter accuracy over meas_n reference cycles
            f_meas += rng.normal(0.0, cfg.fref / (bs.meas_n * np.sqrt(12)))
        bs.observe(f_meas)
    osc.band = bs.band
    return np.asarray(bs.trace, dtype=float)


# ---------------------------------------------------------------- kernels
# The engines' compiled loops (core.jit) take block state as arrays and block
# parameters as scalars, in fixed runs of arguments.  These build those runs
# from the block objects, with the "absent" shape when a block is not wired,
# so every engine spells the hand-off the same way and mypy can follow it.

def osc_law_args(osc_cfg) -> tuple[float, float, float, float, float, int, float, float]:
    """OscConfig.law_params() with its length visible to a type checker: the
    engines' configs are typed Any on the base class, and a star-argument of
    unknown length makes mypy count every argument after it as one too many."""
    return osc_cfg.law_params()


def dtc_kernel_args(dtc_cfg) -> tuple[float, float, int, np.ndarray, bool, float,
                                      float, float, float]:
    """(range_s, t_res, code_max, inl_poly, has_sin, sin_amp, sin_cyc,
    sin_ph, jitter_rms_s) of a DTCConfig, or the inert run for None."""
    if dtc_cfg is None:
        return (0.0, 1.0, 1, np.zeros(0), False, 0.0, 0.0, 0.0, 0.0)
    poly, has_sin, amp, cyc, ph = dtc_cfg.inl_arrays()
    return (float(dtc_cfg.range_s), float(dtc_cfg.t_res),
            (1 << int(dtc_cfg.n_bits)) - 1, poly, has_sin, amp, cyc, ph,
            float(dtc_cfg.jitter_rms_s))


def mash_kernel_args(mash, frac_cfg) -> tuple[bool, int, int, np.ndarray, int]:
    """(has_mash, order, bits, state, frac_word)."""
    if mash is None or frac_cfg is None:
        return (False, 1, 1, np.zeros(8, dtype=np.int64), 0)
    return (True, int(frac_cfg.mash_order), int(frac_cfg.bits), mash.st,
            int(frac_cfg.frac_word))


def cal_kernel_args(cal) -> tuple[bool, int, np.ndarray, float, float, float, float,
                                  bool]:
    """(has_cal, kind, state, mu, mu_final, gear_shift_n, ema, center_err) of
    an LMS gain calibrator; the state array is what the engine loads back."""
    from ..calibration.lms import CAL_STATE
    if cal is None:
        return (False, 0, np.zeros(CAL_STATE), 0.0, float("nan"), -1.0, 0.0, False)
    return (True, int(cal.kind), cal.state(), *cal.params())


def lut_kernel_args(lut_cal, n_cycles: int) -> tuple[bool, np.ndarray, np.ndarray,
                                                    np.ndarray, np.ndarray, float,
                                                    float, int, float, float, int,
                                                    np.ndarray]:
    """(has_lut, lut, counts, xs, state, lo, hi, k, mu, ema, ortho_every,
    snapshots) of a LUT calibrator."""
    if lut_cal is None:
        z = np.zeros(1)
        return (False, z, z, z, np.zeros(2), 0.0, 1.0, 1, 0.0, 0.0, 1,
                np.zeros((0, 1)))
    snaps = np.empty((n_cycles // int(lut_cal.ortho_every) + 1, int(lut_cal.k)))
    return (True, lut_cal.lut, lut_cal.counts, lut_cal._x, lut_cal.state(),
            float(lut_cal.lo), float(lut_cal.hi), int(lut_cal.k), float(lut_cal.mu),
            float(lut_cal._ema), int(lut_cal.ortho_every), snaps)


def fll_kernel_args(fll) -> tuple[bool, np.ndarray, float, float, int, float, float,
                                  float, int]:
    """(has_fll, state, n_target, fref, window, f_engage, f_release, i_fll,
    hyst) of an FLL state machine."""
    if fll is None:
        return (False, np.zeros(5), 0.0, 1.0, 1, 0.0, 1.0, 0.0, 1)
    return (True, fll.st, *fll.params())


def lockdet_kernel_args(det) -> tuple[bool, np.ndarray, float, int, int]:
    """(has_det, state, window_s, count, down_weight) of a lock detector."""
    if det is None:
        return (False, np.zeros(4, dtype=np.int64), 0.0, 1, 0)
    return (True, det.st, float(det.cfg.window_s), int(det.cfg.count),
            int(det.cfg.down_weight))
