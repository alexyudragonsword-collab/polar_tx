# Vendored from pll_simulator@931cfaf: src/pllsim/core/engine.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Reference-edge-driven simulation engine.

Architectures implement step(n) -> dict of recorded signals; the engine owns
the RNG, the main loop, warmup/settle bookkeeping and post-processing of the
recorded phase sequence into a PSD + spur table.
"""
from __future__ import annotations

import numpy as np

from .jit import kernel
from .jitter import rms_jitter_fs
from .results import SimResult
from .spectrum import periodogram_psd, phase_psd


@kernel
def _first_hold(ferr: np.ndarray, tol_hz: float, hold: int) -> int:
    """Index where |ferr| < tol first held for `hold` samples, else -1."""
    run = 0
    for i in range(ferr.shape[0]):
        run = run + 1 if abs(ferr[i]) < tol_hz else 0
        if run >= hold:
            return i - hold + 1
    return -1


def detect_lock(t: np.ndarray, ferr: np.ndarray, tol_hz: float, hold: int = 200) -> float | None:
    """First time |ferr| < tol for `hold` consecutive samples."""
    ferr = np.ascontiguousarray(ferr, dtype=float)
    if ferr.size < hold:
        return None
    i = int(_first_hold(ferr, float(tol_hz), int(hold)))
    return None if i < 0 else float(t[i])


def postprocess(sim: SimResult, settle_frac: float = 0.25,
                int_band: tuple[float, float] = (1e3, 100e6),
                spur_offsets=None, flicker_corner_hz: float = 0.0) -> SimResult:
    """Attach PSD, jitter and spur estimates from the settled portion.

    ``flicker_corner_hz`` is the highest 1/f corner the run was configured
    with (0 = no flicker anywhere); it gates the synthesis-floor note.
    """
    n0 = int(sim.phase_err_out.size * settle_frac)
    ph = sim.phase_err_out[n0:]
    from .boundaries import LOCK_FERR_FRACTION, never_locked, tail_frequency_error
    ferr = tail_frequency_error(sim.freq_out, sim.f0)
    if never_locked(ferr, sim.fs):
        # ILCM/MDLL have no lock detector, and an analog loop's detector is
        # tuned for its design point: without this, a ring railed against
        # its tuning range read as an ordinary result 2.4 GHz off target
        n_tail = max(1, min(5000, len(sim.freq_out) // 4))
        tail = float(np.mean(np.asarray(sim.freq_out[-n_tail:], dtype=float)))
        sim.notes.append(
            f"loop never reached the configured fout: output settled at "
            f"{tail / 1e9:.6f} GHz vs {sim.f0 / 1e9:.6f} GHz configured "
            f"({ferr / 1e6:.3g} MHz off, above fref/{1 / LOCK_FERR_FRACTION:.0f}) "
            "— every figure below describes that unlocked or range-limited "
            "state, not the design. Check osc.f0 and the tuning range "
            "against fout, or run more cycles if it was still acquiring.")
    # A background calibration can still be converging well past settle_frac --
    # a DTC gain LMS that gear-shifts at 100k cycles leaves a fractional spur
    # that dominates everything before it.  The jitter number is then real but
    # answers a different question, so say so rather than let it read as the
    # locked performance.
    half = ph.size // 2
    if half >= 1024:
        # Compared on the cycle-to-cycle increments, not the phase itself: a
        # 1/f wander (a divider or reference flicker corner in a long record)
        # moves the std of one half against the other by up to 2x with no
        # transient anywhere -- measured 0.7-1.95x across seeds on the
        # Dartizio bench once its divider made noise -- while a real
        # calibration transient (the uncalibrated MASH residue at the PD) is
        # high-frequency and shows in the increments at 2x+ (measured 2.2x)
        # against 1.00-1.02 for converged runs of every architecture.
        d = np.diff(ph)
        early, late = float(np.std(d[:half])), float(np.std(d[half:]))
        if early > 1.3 * max(late, 1e-30):
            sim.notes.append(
                f"still settling: cycle-to-cycle phase error is {early / late:.1f}x "
                "larger in the first half of the analysed window than the "
                "second — the jitter below includes an acquisition/calibration "
                "transient. Run more cycles.")
    if ph.size >= 2048:
        from .boundaries import NYQ_FRACTION, jitter_band_clipped
        f_w, s_w = phase_psd(ph, sim.fs)
        sim.f_psd, sim.s_phi_psd = f_w, s_w
        f1 = max(int_band[0], f_w[0])
        f2 = min(int_band[1], NYQ_FRACTION * sim.fs)
        sim.jitter_fs = rms_jitter_fs(f_w, s_w, sim.f0, f1, f2)
        if flicker_corner_hz > 0.0:
            from .boundaries import flicker_floor_hz
            floor = flicker_floor_hz(sim.fs, sim.phase_err_out.size)
            if f1 < floor:
                # only long records get here: the Welch resolution has to
                # drop under fref/65536 first, so no stock run ever did --
                # which is how this boundary sat in the register for a month
                # with a note nothing emitted
                sim.notes.append(
                    f"jitter band starts at {f1:.3g} Hz, below the flicker "
                    f"synthesis floor {floor:.3g} Hz (max(fref/n_settled, "
                    "fref/65536)): synthesized 1/f content below that offset "
                    "is not trustworthy, so neither is the integral over "
                    f"{f1:.3g}..{floor:.3g} Hz — raise the band's lower edge "
                    "or read the figure as approximate there")
        if jitter_band_clipped(int_band[1], sim.fs):
            # the largest silent apples-to-oranges these results carried:
            # analyze() integrates the full band, this number stops at what a
            # record sampled once per reference edge can show, and the two
            # sat side by side in every GUI with nothing saying so
            sim.notes.append(
                f"jitter integrated to {f2 / 1e6:.3g} MHz (0.45x the "
                f"reference-edge record rate), not the full "
                f"{int_band[1] / 1e6:.3g} MHz band the linear model "
                "integrates — compare the two only over the common band")
        if spur_offsets is not None:
            from .spectrum import find_spurs
            f_p, s_p = periodogram_psd(ph, sim.fs)
            sim.spurs_fft = find_spurs(f_p, s_p, spur_offsets)
    return sim
