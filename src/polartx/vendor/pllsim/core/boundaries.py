# Vendored from pll_simulator@5564c64: src/pllsim/core/boundaries.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Where the two domains stop being comparable: the predicates, in one place.

Every function here answers one question of the form "is this operating point
past a structural limit of the time-domain engine or of the linear model?".
They are pure functions on primitives, deliberately: the engines call them to
decide when to append a warning note, and ``pllsim.validation`` wraps the same
functions into the boundary registry the cross-domain sweep asserts against.
One predicate, two consumers -- so the runtime warning and the test contract
cannot drift apart, which is the failure mode that made this module necessary.

The constants are the measured/derived limits themselves.  Change one only
together with the physics that justifies it; the sweep re-measures the
consequences on every CI run.
"""
from __future__ import annotations

#: analyze() for the CPPLL and SPLL is a continuous-time approximation of a
#: sampled loop.  Past UGB = fref/CT_UGB_RATIO the discrete loop's peaking
#: deviates visibly from the s-domain curve (the synthesizer refuses outright
#: at fref/8, synth.py).  The SSPLL and ADPLL use exact z-domain models and
#: are exempt.
CT_UGB_RATIO = 10.0

#: Usable fraction of a sampled record's rate.  Content between 0.45*fs and
#: the Nyquist edge sits in the estimator's roll-off and alias overlap, so
#: spur reporting, jitter integration and PSD comparison all stop here rather
#: than at fs/2 (same constant as arch/base.py's spur filters).
NYQ_FRACTION = 0.45

#: OscPhaseNoiseGen synthesizes 1/f-FM noise in chunks of this many samples
#: (core/colored.py); the sequence decorrelates across refills, so flicker
#: content below fref/FLICKER_CHUNK is not faithfully generated no matter how
#: long the run is.  The other floor is the record itself: nothing below
#: fref/n_settled exists in a finite record (synth_from_psd zeroes DC).
FLICKER_CHUNK = 1 << 16

#: engine.postprocess discards this fraction of the record before the PSD.
SETTLE_FRAC = 0.25


#: An integer-N TDC loop with nothing to dither its TDC parks the phase error
#: inside one bin: the loop cannot see the DCO drift until it crosses a bin
#: edge, and analyze()'s white t_res^2/12 tdc_quant term does not describe
#: that.  The dither that matters is the RMS timing at the TDC input -- its own
#: jitter combined with the loop's phase error, both in seconds -- measured
#: in bins.  Measured 2026-11-12 with the TDC jitter wrapping at the range
#: edge (issue #76's 15 GHz loop and three variants, 400k cycles, seed 3,
#: simulate - analyze averaged over 30-300 kHz): +16.9 / +7.0 / +3.7 /
#: +2.4 / +2.3 / +0.8 dB at 0.08 / 0.21 / 0.26 / 0.47 / 0.33 / 0.41 bins,
#: and -0.3..+0.9 dB at every point from 0.5 bin up.  (Measured the day
#: before, with the jitter still clipped, the jitter rows read up to 4 dB
#: higher below 0.5 bin and up to 10 dB lower above it.)  A frequency ramp or a fractional FCW sweeps the phase
#: across bins and dithers it regardless; analyze() cannot see a ramp, which
#: is why the note it raises says so rather than the predicate.
#:
#: The threshold flags where the white-quantization assumption is
#: unsupported, not a guaranteed error.  The stock adpll_100m_10g moved to
#: integer-N sits at 0.20 bin and measures +0.3 dB (1.6M cycles, every
#: decade 1 kHz-1 MHz within 0.6 dB), while the same loop with a 4x coarser
#: TDC, 0.05 bin, reads +11 to +20 dB.  Dither alone does not predict the
#: size: the DCO's frequency step per code word (20 vs 50 kHz in those two
#: families) sets how fast the parked loop limit-cycles across a bin.  A
#: false warning costs a comparison; a missed 17 dB does not announce itself.
TDC_DITHER_BINS = 0.5


def tdc_dead_zone(fcw: float, t_res: float, dither_rms_s: float) -> bool:
    """Integer-N TDC loop with less than TDC_DITHER_BINS of RMS dither."""
    return abs(fcw - round(fcw)) < 1e-9 and dither_rms_s < TDC_DITHER_BINS * t_res


def ct_approx_exceeded(f_ugb: float, fref: float) -> bool:
    """Linear model past its continuous-time validity (CPPLL/SPLL only)."""
    return f_ugb > fref / CT_UGB_RATIO


def jitter_band_clipped(int_band_hi: float, fs: float) -> bool:
    """simulate() integrates jitter only to NYQ_FRACTION*fs; analyze() does
    the full band.  When this is true the two jitter numbers cover different
    bands and must not be compared directly."""
    return int_band_hi > NYQ_FRACTION * fs


def flicker_floor_hz(fref: float, n_cycles: int) -> float:
    """Lowest offset at which synthesized flicker content is trustworthy."""
    n_settled = max(1, int(n_cycles * (1.0 - SETTLE_FRAC)))
    return max(fref / n_settled, fref / FLICKER_CHUNK)


#: A loop counts as never having reached fout when the mean output frequency
#: over the last LOCK_TAIL_CYCLES sits more than LOCK_FERR_FRACTION*fref away.
#: Measured separation: a truly unlocked or railed loop wanders 1e5..1e9 Hz
#: off, a converged one sits within a few hundred (fref/1000 = 19..250 kHz).
#: Shared by postprocess's runtime note and compare_domains' refusal.
LOCK_FERR_FRACTION = 1e-3
LOCK_TAIL_CYCLES = 5000


def tail_frequency_error(freq_out, fout: float) -> float:
    """|mean of the record's tail - fout| in Hz.

    The tail is the last LOCK_TAIL_CYCLES samples, capped at a quarter of the
    record: a 4000-cycle run that started 5 MHz off and converged to 829 Hz
    read "never reached fout" when the tail was the whole record, transient
    included.
    """
    import numpy as np
    n = len(freq_out)
    n_tail = max(1, min(LOCK_TAIL_CYCLES, n // 4))
    tail = np.asarray(freq_out[-n_tail:], dtype=float)
    return abs(float(np.mean(tail)) - float(fout))


def never_locked(ferr_hz: float, fref: float) -> bool:
    """The output never settled at the configured fout: unlocked, railed
    against a tuning range, or pulled elsewhere.  Every number downstream
    then describes that state, not the design."""
    return ferr_hz > LOCK_FERR_FRACTION * fref


#: Control-voltage travel beyond which no single varactor band is credible.
#: 28-65 nm supplies are 0.9-1.2 V and 180 nm is 1.8 V, so a law that has to
#: move further than this from its free-running point describes an
#: oscillator nobody can build -- the coarse band bank (OscConfig n_bands /
#: band_step_hz) is the physical answer.  Only meaningful when OscConfig sets
#: no v_min/v_max: with a range the law rails and `never_locked` says so.
#: The stock presets sit at 0.6-1.0 V; the GUIs' default -100 MHz hop on a
#: 60 MHz/V oscillator travels 1.67 V, which is why the presets stay
#: unbounded and this is a note rather than a refusal.
TUNING_SWING_V = 1.5


def tuning_law_railed(v_needed: float, v_min, v_max) -> bool:
    """The voltage fout needs lies outside the configured range: the varactor
    rails there, the loop never reaches fout, and the linear model is being
    evaluated at a point the loop cannot occupy."""
    if v_min is not None and v_needed < v_min:
        return True
    return v_max is not None and v_needed > v_max


def tuning_swing_exceeded(v_needed: float, v_min, v_max) -> bool:
    """No range is set and fout needs more than TUNING_SWING_V of travel from
    f0: the unbounded law follows it, a real oscillator would not."""
    if v_min is not None or v_max is not None:
        return False
    return abs(v_needed) > TUNING_SWING_V


def conditionally_stable(n_crossings: int) -> bool:
    """More than one gain crossover: phase margin read at one crossing does
    not describe the loop, and the linear jitter integral spans a region
    where the loop is not small-signal stable."""
    return n_crossings != 1
