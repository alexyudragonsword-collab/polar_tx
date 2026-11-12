# Vendored from pll_simulator@5564c64: src/pllsim/arch/adpll.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""All-digital PLL.

Two variants:
  mode="tdc":      counter-assisted (Staszewski) — FCW accumulator vs DCO
                   counter + flash TDC fractional readout; supports fractional
                   FCW natively (no MASH needed).
  mode="dtc_bbpd": divider + MASH + DTC-aligned bang-bang PD — the common
                   low-power fractional-N style; BBPD gain linearized
                   self-consistently in analyze().

Frequency domain is exact z-domain on the grid (z = e^{j2πf/fref}) including
the one-cycle update delay.  Time domain runs one step per reference cycle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..blocks.dtc import dtc_code, dtc_inl_s, dtc_time
from ..blocks.oscillator import OscConfig, Oscillator, osc_freq_law
from ..blocks.tdc import (
    BBPD,
    TDC,
    TDCConfig,
    bbpd_decide,
    meta_gain_penalty,
    tdc_measure,
)
from ..calibration.gain_cal import kdco_perturbation, kdco_step, tdc_period_step
from ..calibration.lms import lms_step
from ..core.colored import synth_from_psd
from ..core.deltasigma import mash_residual, mash_step
from ..core.dtcspurs import frac_spur_offsets
from ..core.engine import detect_lock, postprocess
from ..core.freqresp import FreqResponse, default_grid, loop_metrics
from ..core.jit import kernel
from ..core.jitter import integrate_pn, ipn_dbc, rms_jitter_fs
from ..core.noise import (
    DcoQuantPhase,
    FlickerFloorPhase,
    NoisePath,
    NoiseSource,
    ShapedQuantization,
    output_psd,
)
from ..core.results import AnalysisResult, SimResult
from .base import (
    PLLBase,
    add_pull_offset,
    cal_kernel_args,
    dtc_kernel_args,
    dtc_t_target_of,
    flicker_corner_hz,
    osc_law_args,
    pull_hz,
    pull_notes,
    pull_spur,
    supply_ripple_v,
)
from .frac import FracConfig  # polartx: extracted from cppll.py

TWOPI = 2.0 * np.pi


@dataclass
class DLFConfig:
    """Digital loop filter: L(z) = alpha + rho/(1-z^-1), then IIR stages
    lam/(1-(1-lam) z^-1) each."""

    alpha: float
    rho: float
    iir_lambdas: tuple = ()

    def __post_init__(self):
        if self.alpha < 0 or self.rho < 0 or not (self.alpha or self.rho):
            raise ValueError("DLFConfig alpha and rho are non-negative and not "
                             f"both zero, got alpha={self.alpha}, rho={self.rho}")
        if any(not 0.0 < lam <= 1.0 for lam in self.iir_lambdas):
            raise ValueError("DLFConfig iir_lambdas are IIR coefficients in "
                             f"(0, 1], got {self.iir_lambdas}")


@dataclass
class ADPLLConfig:
    fref: float
    fout: float
    osc: OscConfig                     # gain = Kdco [Hz/LSB]
    dlf: DLFConfig
    mode: str = "tdc"                  # "tdc" | "dtc_bbpd"
    tdc: TDCConfig | None = None
    dco_dither_order: int = 1          # 0 = none, 1 = EFM1 at fref
    ref_pn_dbchz: float = -155.0
    ref_pn_fc: float = 20e3
    kdco_est_error: float = 0.0        # initial Kdco estimate error (fraction)
    frac: FracConfig | None = None     # dtc_bbpd mode: MASH + DTC config
    bb_jitter_rms_s: float = 100e-15   # BBPD input-referred jitter (dtc_bbpd)
    # Sampling-flop resolution window: inside it the decision is a coin flip.
    # Costs Kbb by exp(-W^2/2 sigma^2) and therefore widens the input-referred
    # noise by the same factor -- a window equal to sigma_t is 4.34 dB.
    bb_meta_window_s: float = 0.0
    # Feedback divider phase noise (dtc_bbpd mode only: the TDC path reads the
    # DCO edge against the reference and has no divider).  None = not
    # modelled, which analyze() says out loud; the CPPLL/SPLL carry the same
    # pair with -160 dBc/Hz / 100 kHz defaults, and the ADPLL had no term at
    # all until the health check listed it.
    div_pn_dbchz: float | None = None
    div_pn_fc: float | None = None
    int_band: tuple[float, float] = (1e3, 100e6)

    @property
    def fcw(self) -> float:
        return self.fout / self.fref

    def __post_init__(self):
        if self.mode not in ("tdc", "dtc_bbpd"):
            raise ValueError("mode must be 'tdc' or 'dtc_bbpd'")
        if self.mode == "tdc" and self.tdc is None:
            raise ValueError("tdc mode requires TDCConfig")
        if self.mode == "dtc_bbpd" and self.frac is None:
            raise ValueError("dtc_bbpd mode requires FracConfig (MASH + DTC)")
        if self.mode == "tdc" and (self.div_pn_dbchz is not None
                                   or self.div_pn_fc is not None):
            raise ValueError(
                "tdc mode has no feedback divider (the TDC reads the DCO edge "
                "against the reference): div_pn_dbchz / div_pn_fc would be "
                "accepted and ignored -- leave them unset")
        if (self.div_pn_dbchz is None) != (self.div_pn_fc is None):
            raise ValueError(
                "div_pn_dbchz and div_pn_fc describe one divider: set both "
                f"or neither (got {self.div_pn_dbchz}, {self.div_pn_fc})")
        if self.frac is not None and \
                abs((self.fout / self.fref) % 1.0 - self.frac.frac) > 1e-6:
            raise ValueError(
                f"fout/fref fractional part {(self.fout / self.fref) % 1.0:.6f} "
                f"does not match FracConfig.frac {self.frac.frac}: the divider "
                f"locks at (n_int + frac)*fref, not at fout -- change fout or "
                f"frac together with fref")
        if self.frac is not None and self.frac.dtc_lut_cal is not None:
            raise ValueError(
                "dtc_lut_cal is wired for the CPPLL only: its update"
                " regresses a TIMING error against the MASH residue, and"
                " the bang-bang detector exposes only a sign — wiring it"
                " naively measurably made the INL spur worse.")
        if self.osc.nl1 or self.osc.nl2:
            # the DCO word is an integer in LSB, thousands of them at the
            # operating point, and analyze() uses the linear Kdco -- a per-volt
            # nonlinearity applied to that word is not a DCO model.  Found by
            # the field-sensitivity gate: nl2 = 1e-3 drove the frequency law to
            # NaN inside the TDC.  Refused, like the ILCM does, rather than
            # accepted and numerically wrong.
            raise ValueError(
                "ADPLL does not model DCO nonlinearity: OscConfig nl1/nl2 are "
                "per-volt coefficients and the DCO word is in LSB; analyze() "
                "uses the linear Kdco and the engine would compute garbage")
        if self.mode == "dtc_bbpd" and self.kdco_est_error:
            # the BBPD loop filter drives the DCO word directly (LSB in, LSB
            # out) -- there is no Kdco estimate anywhere in that path for an
            # error to live in.  The field read correctly and did nothing.
            raise ValueError(
                "kdco_est_error applies to the counter/TDC loop only: the "
                "dtc_bbpd loop filter drives the DCO word directly and never "
                "normalises by a Kdco estimate")
        if self.osc.n_bands > 1:
            raise ValueError(
                "the ADPLL engine has no coarse-band search: the DCO word is\n"
                "the only tuning control it models, so n_bands would be\n"
                "accepted and never acted on (and export would still emit a\n"
                "band-search FSM for it)")


@kernel
def adpll_tdc_kernel(n_cycles: int, tref: float, fout: float, fref: float, fcw: float,
                     f0: float, gain: float, nl1: float, nl2: float, band_step: float,
                     n_bands: int, v_lo: float, v_hi: float, pushing: float,
                     v_sup: np.ndarray, f_pull: np.ndarray, osc_noise: np.ndarray,
                     jit_ref: np.ndarray, noise_on: bool, calibration: bool,
                     zn: np.ndarray,
                     tdc_t_lsb_true: float, tdc_code_max: int, tdc_jitter: float,
                     tdc_has_sin: bool, tdc_sin_amp: float, tdc_sin_cyc: float,
                     tdc_sin_ph: float, cpp_nom: float,
                     has_tcal: bool, tcal_st: np.ndarray, tcal_ema: float,
                     tcal_trace: np.ndarray,
                     has_kcal: bool, kcal_st: np.ndarray, kcal_halves: np.ndarray,
                     kcal_ests: np.ndarray, kcal_amp: float, kcal_meas_n: int,
                     kcal_rounds: int, kcal_settle: int, kcal_trace: np.ndarray,
                     kdco_hat0: float, otw_center: float, dlf_alpha: float,
                     dlf_rho: float, iir_lambdas: np.ndarray, iir_state: np.ndarray,
                     dither: bool, has_mod: bool, mod_freq: np.ndarray, mod_dp_gain: float,
                     mod_freq_dp: np.ndarray, has_dpcal: bool, dpcal_step: Any,  # polartx
                     dpcal_st: np.ndarray, dpcal_pr: np.ndarray,  # polartx
                     dpcal_rec: np.ndarray, dp_trace: np.ndarray,  # polartx
                     phase_err: np.ndarray, freq_out: np.ndarray, otw_rec: np.ndarray
                     ) -> int:
    """Counter-assisted (Staszewski) loop, one reference cycle per iteration.
    Returns the normal draws consumed."""
    kdco_hat = kdco_hat0
    acc = 0.0
    qerr = 0.0                       # 1st-order DCO dither state
    phi_v = 0.0                      # DCO phase [UI]
    r_acc = 0.0                      # reference accumulator [UI]
    prev_phi_n = 0.0
    zi = 0
    fv = osc_freq_law(otw_center, 0, f0, gain, nl1, nl2, band_step, n_bands, v_lo, v_hi)
    n_iir = iir_lambdas.shape[0]
    g_dp = mod_dp_gain               # polartx: direct-point gain, dp_cal-updated
    for n in range(n_cycles):
        d_phi_n = (osc_noise[n] - prev_phi_n) / TWOPI     # UI
        prev_phi_n = osc_noise[n]
        phi_v += fv * tref + d_phi_n
        r_acc += fcw
        if has_mod:
            # lowpass point [UI] — one cycle behind the direct point:
            # the fv used for phi_v this cycle carries mod_freq[n-1]
            r_acc += (mod_freq[n - 1] if n > 0 else 0.0) * tref
        # reference jitter shifts the sampling instant of the DCO phase
        phi_sample = phi_v + jit_ref[n] * fv
        if noise_on and tdc_jitter > 0:
            # TDC jitter is input-referred, so it shifts the same instant:
            # count and fraction move together and a sample pushed across a
            # DCO edge wraps to it.  It used to be added to the fraction
            # after the count and clamped at 0, which cut part of it at the
            # integer-N lock point, where the TDC sits at its range edge
            # (issue #76: -5..-6 dB with the jitter dominant).
            phi_sample = phi_sample + tdc_jitter * zn[zi] * fv
            zi += 1
        count = float(math.floor(phi_sample))
        frac_ui = phi_sample - count
        tdco = 1.0 / fv
        dt_meas = frac_ui * tdco
        code = tdc_measure(dt_meas, tdc_t_lsb_true, tdc_code_max, tdc_has_sin,
                           tdc_sin_amp, tdc_sin_cyc, tdc_sin_ph)
        if has_tcal and calibration:
            cpp_meas = tdco / tdc_t_lsb_true
            if noise_on:
                cpp_meas = cpp_meas + 0.5 * zn[zi]
                zi += 1
            tcal_trace[n] = tdc_period_step(tcal_st, cpp_meas, tcal_ema)
            tdc_ui = code / tcal_st[0]
        else:
            tdc_ui = code / cpp_nom
        e_ui = r_acc - (count + tdc_ui)

        if has_kcal and calibration and kcal_st[3] == 0.0:
            # open-loop FCAL phase: loop frozen, OTW stepped +/-A; the
            # frequency is measured from the counter (phase slope)
            kcal_trace[n] = kdco_step(kcal_st, kcal_halves, kcal_ests, fv, kcal_amp,
                                      kcal_meas_n, kcal_rounds, kcal_settle)
            kdco_hat = kcal_st[4]
            otw = otw_center + kdco_perturbation(kcal_st, kcal_amp, kcal_meas_n)
            r_acc = count + tdc_ui       # keep PD aligned for loop closure
            acc = 0.0
        else:
            # DLF
            acc += dlf_rho * e_ui
            x = dlf_alpha * e_ui + acc
            for i in range(n_iir):
                iir_state[i] += iir_lambdas[i] * (x - iir_state[i])
                x = iir_state[i]
            otw = x * fref / kdco_hat + otw_center
        if has_dpcal and calibration:            # polartx: online dp_cal
            g_dp = dpcal_step(dpcal_st, dpcal_pr, e_ui, mod_freq[n], dpcal_rec, n)
            dp_trace[n] = g_dp
        if has_mod:                              # highpass (direct) point
            # polartx: the direct point's own trajectory (mod_freq itself
            # unless a range-limited direct DAC clips it) at the gain dp_cal
            # tracks
            otw += mod_freq_dp[n] * g_dp / kdco_hat
        # DCO quantization with optional 1st-order dither
        if dither:
            otw_q = float(math.floor(otw + qerr))
            qerr = otw + qerr - otw_q
        else:
            otw_q = float(round(otw))
        fv = (osc_freq_law(otw_q, 0, f0, gain, nl1, nl2, band_step, n_bands, v_lo, v_hi)
              + pushing * v_sup[n] + f_pull[n])

        phase_err[n] = TWOPI * (phi_v - (n + 1) * fcw)
        freq_out[n] = fv
        otw_rec[n] = otw_q
    return zi



# ------------------------------------------------------------------ polartx
# Online two-point gain calibration (dp_cal) inside the kernel.  The
# calibrator runs in the loop through a step kernel over a small state
# vector, the way the TDC-period and Kdco calibrators above do:
#
#     g = step(st, pr, err_ui, reg_hz, rec, n)
#
# ``st`` is its state (updated in place), ``pr`` its parameters, ``rec`` a
# per-cycle record it may fill (one row per cycle).  A calibrator hands over
# its step with ``kernel_view(n_cycles) -> (step, st, pr, rec)`` and takes
# its state back with ``kernel_done(st, rec, n_steps)``; the LMS gain
# calibrators of calibration.lms are adapted below.  It used to be a Python
# object called back every cycle, which a compiled loop cannot do.


@kernel
def dp_cal_off(st: np.ndarray, pr: np.ndarray, err: float, reg: float,
               rec: np.ndarray, n: int) -> float:
    """The step passed when no calibrator is attached; never called."""
    return st[0]


@kernel
def lms_dp_step(st: np.ndarray, pr: np.ndarray, err: float, reg: float,
                rec: np.ndarray, n: int) -> float:
    """A calibration.lms gain calibrator as a dp_cal step.  ``pr`` is
    (kind, mu, mu_final, gear_shift_n, ema, center_err)."""
    return lms_step(int(pr[0]), st, err, reg, pr[1], pr[2], pr[3], pr[4],
                    pr[5] != 0.0)


def _dp_cal_view(dp_cal: Any, n_cycles: int) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    if hasattr(dp_cal, "kernel_view"):
        return dp_cal.kernel_view(n_cycles)
    if all(hasattr(dp_cal, a) for a in ("kind", "state", "params", "load_state")):
        pr = np.array([float(dp_cal.kind)] + [float(x) for x in dp_cal.params()])
        return lms_dp_step, dp_cal.state(), pr, np.zeros((0, 0))
    raise TypeError(
        "dp_cal must be a calibration.lms gain calibrator or expose "
        "kernel_view(n_cycles) / kernel_done(st, rec, n_steps)")


def _dp_cal_done(dp_cal: Any, st: np.ndarray, rec: np.ndarray,
                 trace: np.ndarray) -> None:
    if hasattr(dp_cal, "kernel_done"):
        dp_cal.kernel_done(st, rec, trace.size)
    else:
        dp_cal.load_state(st)
        dp_cal.trace.extend(trace.tolist())

@kernel
def adpll_bbpd_kernel(n_cycles: int, tref: float, fout: float, n_int: int,
                      f0: float, gain: float, nl1: float, nl2: float, band_step: float,
                      n_bands: int, v_lo: float, v_hi: float, pushing: float,
                      v_sup: np.ndarray, f_pull: np.ndarray, osc_noise: np.ndarray,
                      jit_ref: np.ndarray, jit_div: np.ndarray, noise_on: bool,
                      zn: np.ndarray, mash_order: int, mash_bits: int,
                      mash_st: np.ndarray, frac_word: int,
                      has_dtc: bool, dtc_range_s: float, dtc_t_res: float,
                      dtc_code_max: int, dtc_inl_poly: np.ndarray, dtc_has_sin: bool,
                      dtc_sin_amp: float, dtc_sin_cyc: float, dtc_sin_ph: float,
                      dtc_jitter: float, dtc_gain_error: float, has_drift: bool,
                      dtc_gain_drift: np.ndarray,
                      has_cal: bool, cal_kind: int, cal_st: np.ndarray, cal_mu: float,
                      cal_mu_final: float, cal_gear: float, cal_ema: float,
                      cal_center: bool, cal_trace: np.ndarray,
                      bb_jitter: float, bb_meta: float, otw_center: float,
                      dlf_alpha: float, dlf_rho: float, dither: bool,
                      phase_err: np.ndarray, freq_out: np.ndarray, otw_rec: np.ndarray
                      ) -> tuple[int, int]:
    """Divider + MASH + DTC-aligned bang-bang loop, one reference cycle per
    iteration.  Returns (metastable decisions, normal draws consumed)."""
    acc = 0.0
    qerr = 0.0
    phi_out = 0.0
    t_div = 0.0
    prev_phi_n = 0.0
    gain_corr = 1.0
    gain_err = dtc_gain_error
    n_meta = 0
    zi = 0
    fv = osc_freq_law(otw_center, 0, f0, gain, nl1, nl2, band_step, n_bands, v_lo, v_hi)
    for n in range(n_cycles):
        t_ref = n * tref + jit_ref[n]
        residual_ui = mash_residual(mash_bits, mash_st)
        if has_dtc:
            if has_drift:
                gain_err = dtc_gain_drift[n]
            code = dtc_code(residual_ui / fout, dtc_range_s, gain_corr, dtc_t_res,
                            dtc_code_max)
            inl = dtc_inl_s(code, dtc_code_max, dtc_inl_poly, dtc_has_sin, dtc_sin_amp,
                            dtc_sin_cyc, dtc_sin_ph)
            d = dtc_time(code, dtc_t_res, gain_err, inl)
            if noise_on and dtc_jitter > 0:
                d += dtc_jitter * zn[zi]
                zi += 1
            t_ref += d
        # the divider's own edge jitter is sampled at the PD and does not
        # accumulate in the count (same as the CPPLL's jit_div)
        dt = t_div + jit_div[n] - t_ref
        if noise_on and bb_jitter > 0:
            dt = dt + bb_jitter * zn[zi]
            zi += 1
        if bb_meta > 0 and abs(dt) < bb_meta:
            n_meta += 1
            if noise_on:
                # a coin flip: the sign of the next draw of the pool
                e = 1 if zn[zi] < 0.0 else -1
                zi += 1
            else:
                e = bbpd_decide(dt)
        else:
            e = bbpd_decide(dt)

        if has_cal and has_dtc:
            gain_corr = lms_step(cal_kind, cal_st, float(e), residual_ui, cal_mu,
                                 cal_mu_final, cal_gear, cal_ema, cal_center)
            cal_trace[n] = gain_corr

        acc += dlf_rho * e
        otw = dlf_alpha * e + acc + otw_center
        if dither:
            otw_q = float(math.floor(otw + qerr))
            qerr = otw + qerr - otw_q
        else:
            otw_q = float(round(otw))
        fv = (osc_freq_law(otw_q, 0, f0, gain, nl1, nl2, band_step, n_bands, v_lo, v_hi)
              + pushing * v_sup[n] + f_pull[n])

        n_next = n_int + mash_step(mash_order, mash_bits, mash_st, frac_word)
        d_osc = osc_noise[n] - prev_phi_n
        prev_phi_n = osc_noise[n]
        t_div += n_next / fv - d_osc / (TWOPI * fv)

        phi_out += TWOPI * (fv - fout) * tref + d_osc
        phase_err[n] = phi_out
        freq_out[n] = fv
        otw_rec[n] = otw_q
    return n_meta, zi


class ADPLL(PLLBase):
    def __init__(self, cfg: ADPLLConfig):
        self.cfg = cfg

    # ------------------------------------------------------------- helpers
    def _dlf_fr(self, f: np.ndarray) -> FreqResponse:
        c = self.cfg.dlf
        zinv = np.exp(-2j * np.pi * f / self.cfg.fref)
        h = c.alpha + c.rho / (1.0 - zinv)
        for lam in c.iir_lambdas:
            h = h * lam / (1.0 - (1.0 - lam) * zinv)
        return FreqResponse(f, h)

    def _dco_phase_fr(self, f: np.ndarray) -> FreqResponse:
        """OTW [LSB] -> output phase [rad]: 2π·Kdco·Tref·z^-1/(1-z^-1)."""
        c = self.cfg
        zinv = np.exp(-2j * np.pi * f / c.fref)
        return FreqResponse(f, TWOPI * c.osc.gain * (1.0 / c.fref) * zinv / (1.0 - zinv))

    # ------------------------------------------------------------- analyze
    def tdc_dither_rms_s(self) -> float:
        """RMS timing [s] at the TDC input that can carry it across bins.

        The TDC's own jitter combined with the loop's phase error, the
        latter integrated from 1 kHz to fref/2 with the three quantization
        terms taken out: tdc_quant and tdc_jitter are what the dither acts on,
        and dco_quant stops being random once the TDC code stops changing.
        This is the quantity ``core.boundaries.tdc_dead_zone`` compares with
        one bin.
        """
        c = self.cfg
        fg = np.geomspace(1e3, c.fref / 2.0, 400)
        _gol, _kdet, _h, paths, _notes = self._noise_paths(fg)
        bd = output_psd(paths, fg)
        s = (bd["total"] - bd.get("tdc_quant", 0.0) - bd.get("tdc_jitter", 0.0)
             - bd.get("dco_quant", 0.0))
        sigma_loop_s = math.sqrt(max(integrate_pn(fg, s, fg[0], fg[-1]), 0.0)) \
            / (TWOPI * c.fout)
        return math.hypot(c.tdc.jitter_rms_s, sigma_loop_s)

    def _noise_paths(self, f: np.ndarray):
        """(gol, kdet, h, paths, notes) on grid ``f``: the noise model alone."""
        c = self.cfg
        notes: list[str] = []
        if c.mode == "tdc":
            gol, kdet = self._gol_tdc(f)
        else:
            gol, kdet, sigma_t = self._gol_bbpd(f)
            notes.append(f"BBPD linearized: sigma_t at PD = {sigma_t * 1e15:.0f} fs, "
                         f"Kbb = {kdet:.3g} 1/s")
        h = gol.feedback()               # lowpass, ref-referred
        err = 1.0 - h

        paths = [
            NoisePath(FlickerFloorPhase.from_spot("ref", c.ref_pn_dbchz, c.ref_pn_fc),
                      h * c.fcw),
            NoisePath(c.osc.leeson("dco"), err),
            NoisePath(DcoQuantPhase(name="dco_quant", kdco=c.osc.gain, fs=c.fref,
                                    order=max(c.dco_dither_order, 0)), err),
        ]
        if c.mode == "tdc":
            q_ui = c.tdc.t_res * c.fout
            paths.append(NoisePath(
                ShapedQuantization(name="tdc_quant", unit="rad^2/Hz",
                                   q=TWOPI * q_ui, fs=c.fref, order=0), h))
            if c.tdc.jitter_rms_s > 0:
                # TDC.measure adds it to the interval once per reference
                # cycle, before quantizing, so it is a sampled white source on
                # the same path as tdc_quant and in the same convention:
                # 2 sigma^2 / fref (tdc_quant is the sigma = q/sqrt(12) case).
                # Until 2026-11 only simulate() read this field (issue #76).
                paths.append(NoisePath(
                    NoiseSource(name="tdc_jitter", unit="rad^2/Hz",
                                level=2.0 * (TWOPI * c.fout * c.tdc.jitter_rms_s) ** 2
                                / c.fref), h))
        else:
            if c.div_pn_dbchz is not None:
                # divider output phase enters at the PD like the reference:
                # low-passed, multiplied up by the ratio
                paths.append(NoisePath(
                    FlickerFloorPhase.from_spot("divider", c.div_pn_dbchz,
                                                c.div_pn_fc), h * c.fcw))
            else:
                notes.append("divider phase noise not modelled (div_pn_dbchz "
                             "unset): the feedback divider is noiseless here")
            # BBPD quantization: total power (1 - 2/pi) white, input-referred
            s_bb = 2.0 * (1.0 - 2.0 / np.pi) / c.fref / kdet**2   # s^2/Hz
            paths.append(NoisePath(
                NoiseSource(name="bbpd_quant", unit="rad^2/Hz",
                            level=s_bb * (TWOPI * c.fout) ** 2), h))
            if c.frac.dtc is not None:
                q_dtc = c.frac.dtc.t_res
                paths.append(NoisePath(
                    ShapedQuantization(name="dtc_quant", unit="rad^2/Hz",
                                       q=TWOPI * q_dtc * c.fout, fs=c.fref, order=0), h))
                j_dtc = getattr(c.frac.dtc, "jitter_rms_s", 0.0)
                if j_dtc > 0:      # DTC.delay injects it every cycle in sim
                    paths.append(NoisePath(
                        NoiseSource(name="dtc_jitter", unit="rad^2/Hz",
                                    level=2.0 * (TWOPI * c.fout * j_dtc) ** 2
                                    / c.fref), h))
                eps = getattr(c.frac.dtc, "gain_error_residual", 0.01)
                paths.append(NoisePath(
                    ShapedQuantization(name="dsm_residual", unit="rad^2/Hz",
                                       q=TWOPI * eps, fs=c.fref,
                                       order=c.frac.mash_order - 1), h))
        return gol, kdet, h, paths, notes

    def analyze(self, f: np.ndarray | None = None) -> AnalysisResult:
        c = self.cfg
        if f is None:
            f = default_grid(1e2, 1e9)
        gol, kdet, h, paths, notes = self._noise_paths(f)
        err = 1.0 - h

        m = loop_metrics(gol, f_limit=c.fref / 2)
        bd = output_psd(paths, f)
        jit = rms_jitter_fs(f, bd["total"], c.fout, *c.int_band)
        if c.mode == "tdc":
            from ..core.boundaries import TDC_DITHER_BINS, tdc_dead_zone
            dither_s = self.tdc_dither_rms_s()
            if tdc_dead_zone(c.fcw, c.tdc.t_res, dither_s):
                notes.append(
                    f"TDC dead zone: integer-N (FCW {c.fcw:g}) with "
                    f"{dither_s / c.tdc.t_res:.2f} TDC bins of RMS dither at "
                    f"its input, under {TDC_DITHER_BINS:g}.  The loop parks "
                    "inside one bin and cannot see the DCO drift there, so "
                    "tdc_quant's white t_res^2/12 is not supported: "
                    "simulate() can read up to ~20 dB more below the loop "
                    "bandwidth (measured +16.9 dB at 0.08 bin, but +0.3 dB "
                    "for the stock loop at 0.20 bin -- the size depends on "
                    "the loop, so compare with simulate()).  A "
                    "frequency ramp or two-point modulation dithers the TDC "
                    "and this does not apply; neither does it in fractional-N."
                    "  TDC jitter of ~0.5 bin or more also dithers it.")
        if m.f_ugb > c.fref / 10:
            notes.append("UGB > fref/10: discrete loop peaking significant")
        from ..core.boundaries import conditionally_stable
        if conditionally_stable(m.n_crossings):
            notes.append(
                f"open loop crosses unity gain {m.n_crossings}x below "
                "fref/2: conditionally stable — phase margin at the first "
                "crossing does not describe the loop")
        spurs = dict(pull_spur(c.osc, err))
        notes.extend(pull_notes(c.osc))
        if c.mode == "tdc" and c.fcw % 1.0 > 1e-9:
            # The TDC input sweeps its range at the fractional beat rate, so a
            # declared INL is a deterministic tone generator, not something
            # only simulate() can see.
            from ..core.tdcspurs import code_span, tdc_inl_spur_table
            for off, dbc in tdc_inl_spur_table(
                    c.tdc, c.fcw % 1.0, c.fref, c.fout, ntf=h).items():
                spurs[f"frac_spur@{off:.0f}Hz"] = dbc
            span = code_span(c.tdc, c.fout)
            if span > 1.0:
                notes.append(
                    f"TDC range is {1 / span:.2f} of an output period: it "
                    "saturates every cycle, so the loop loses phase "
                    "information rather than merely quantizing it")
            if not c.tdc.inl_sin:
                offs = frac_spur_offsets(c.fcw % 1.0, c.fref)
                where = (f"{min(offs) / 1e6:.3f} MHz and {len(offs) - 1} more"
                         if offs else "fold(k*frac)*fref")
                notes.append(
                    f"fractional FCW with an ideal TDC: the beats at {where} "
                    "come from quantization alone, which simulate() measures "
                    "— declare tdc.inl_sin to get them predicted here")
        if c.mode == "dtc_bbpd" and c.frac.dtc is not None:
            from ..core.dtcspurs import dtc_spur_table
            eps = getattr(c.frac.dtc, "gain_error_residual", 0.01)
            for off, dbc in dtc_spur_table(
                    c.frac, dtc_t_target_of(self), c.fref, c.fout,
                    ntf=h, gain_eps=eps).items():
                spurs[f"frac_spur@{off:.0f}Hz"] = dbc
        return AnalysisResult(
            f=f, f0=c.fout, pn_breakdown=bd, loop=m, jitter_fs=jit,
            ipn_dbc=ipn_dbc(f, bd["total"], *c.int_band), int_band=c.int_band,
            spurs_analytic=spurs,
            ntfs={"gol": gol, "h": h, "err": err}, notes=notes)

    def _gol_tdc(self, f):
        # loop works in UI: e_ui -> L(z) -> x fref/kdco_hat -> OTW -> DCO phase
        # Gol(z) = L(z) * (kdco_true/kdco_hat) * z^-1/(1-z^-1)
        c = self.cfg
        zinv = np.exp(-2j * np.pi * f / c.fref)
        kr = 1.0 / (1.0 + c.kdco_est_error)
        gol = self._dlf_fr(f) * FreqResponse(f, kr * zinv / (1.0 - zinv))
        return gol, None

    def _gol_bbpd(self, f):
        """Self-consistent BBPD linearization: Kbb = sqrt(2/pi)/sigma_t."""
        c = self.cfg
        q_dtc = c.frac.dtc.t_res if c.frac.dtc is not None else 0.0
        j_dtc = c.frac.dtc.jitter_rms_s if c.frac.dtc is not None else 0.0
        sigma_t = max(c.bb_jitter_rms_s, 1e-15)
        gol = None
        for _ in range(8):
            # a metastability window costs gain, not output power: the flop
            # still emits +/-1, just uncorrelated with dt inside the window
            kbb = (np.sqrt(2.0 / np.pi) / sigma_t
                   * meta_gain_penalty(c.bb_meta_window_s, sigma_t))
            # e = Kbb * dt_pd ; dt_pd = phi_out/(2π fout)
            det = kbb / (TWOPI * c.fout)
            gol = self._dlf_fr(f) * self._dco_phase_fr(f) * det
            h = gol.feedback()
            err = 1.0 - h
            # output jitter contributions at the PD
            s_dco = c.osc.leeson("dco").psd(f) * err.mag2()
            from ..core.jitter import integrate_pn
            var_phi = integrate_pn(f, s_dco, 1e4, c.fref / 2)
            # everything the comparator actually sees: loop-shaped DCO noise,
            # DTC quantization AND the DTC's own random jitter (it sits in the
            # same edge path, so leaving it out over-predicts Kbb and the BW)
            var_t = var_phi / (TWOPI * c.fout) ** 2 \
                + q_dtc**2 / 12.0 + j_dtc**2 + c.bb_jitter_rms_s**2
            sigma_new = np.sqrt(var_t)
            if abs(sigma_new - sigma_t) < 1e-18:
                break
            sigma_t = 0.5 * sigma_t + 0.5 * sigma_new
        kbb = (np.sqrt(2.0 / np.pi) / sigma_t
               * meta_gain_penalty(c.bb_meta_window_s, sigma_t))
        return gol, kbb, sigma_t

    # ------------------------------------------------------------ simulate
    def simulate(self, n_cycles: int, *, noise: bool = True, calibration: bool = True,
                 seed: int = 0, f_start_offset: float = 0.0,
                 kdco_cal=None, tdc_cal=None, dtc_gain_init_error: float = 0.0,
                 supply_ripple: tuple[float, float] | None = None,
                 mod_freq: np.ndarray | None = None, mod_dp_gain: float = 1.0,
                 dtc_gain_drift: np.ndarray | None = None,
                 mod_freq_dp: np.ndarray | None = None,
                 dp_cal: Any = None) -> SimResult:
        """The two modes take different knobs, and asking for the wrong one
        raises rather than being ignored: a silently dropped mod_freq returns a
        perfectly normal-looking SimResult with no modulation in it, and the
        EVM then reads as noise-limited.

        mode="tdc":      kdco_cal, tdc_cal, mod_freq, mod_dp_gain
        mode="dtc_bbpd": dtc_gain_init_error, dtc_gain_drift
        Both: noise, calibration, seed, f_start_offset.

        polartx extensions, mode="tdc" only:
        mod_freq_dp: a separate trajectory for the direct point -- a
        range-limited direct-modulation DAC (the FCW path keeps the full
        mod_freq).  None = mod_freq.
        dp_cal: a background two-point gain calibrator stepped every cycle
        from the phase error against mod_freq; its value REPLACES
        mod_dp_gain, its trace lands in cal_traces['dp_gain'].  A
        calibration.lms gain calibrator, or anything exposing
        kernel_view/kernel_done (see the dp_cal section above).
        """
        if self.cfg.mode == "tdc":
            self._reject("tdc", dtc_gain_init_error=dtc_gain_init_error,
                         dtc_gain_drift=dtc_gain_drift)
            return self._sim_tdc(n_cycles, noise, calibration, seed,
                                 f_start_offset, kdco_cal, tdc_cal,
                                 mod_freq, mod_dp_gain, supply_ripple,
                                 mod_freq_dp, dp_cal)
        self._reject("dtc_bbpd", kdco_cal=kdco_cal, tdc_cal=tdc_cal,
                     mod_freq=mod_freq, mod_dp_gain=None
                     if mod_dp_gain == 1.0 else mod_dp_gain,
                     mod_freq_dp=mod_freq_dp, dp_cal=dp_cal)  # polartx
        return self._sim_bbpd(n_cycles, noise, calibration, seed,
                              f_start_offset, dtc_gain_init_error,
                              dtc_gain_drift, supply_ripple)

    @staticmethod
    def _reject(mode: str, **unsupported):
        for name, value in unsupported.items():
            if value is None or (np.isscalar(value) and value == 0.0):
                continue
            raise TypeError(
                f"ADPLL.simulate(): {name} is not supported in mode={mode!r} "
                "and would have been ignored")

    def _ref_jitter(self, n_cycles, rng, noise):
        c = self.cfg
        if not noise:
            return np.zeros(n_cycles)
        src = FlickerFloorPhase.from_spot("ref", c.ref_pn_dbchz, c.ref_pn_fc)
        return synth_from_psd(src.psd, c.fref, n_cycles, rng) / (TWOPI * c.fref)

    def _div_jitter(self, n_cycles, rng, noise):
        """Divider-output edge jitter [s] per reference cycle, the same
        synthesis the CPPLL uses for its non-retimed divider."""
        c = self.cfg
        if not noise or c.div_pn_dbchz is None:
            return np.zeros(n_cycles)
        src = FlickerFloorPhase.from_spot("div", c.div_pn_dbchz, c.div_pn_fc)
        return synth_from_psd(src.psd, c.fref, n_cycles, rng) / (TWOPI * c.fref)

    def _sim_tdc(self, n_cycles, noise, calibration, seed, f_start_offset,
                 kdco_cal, tdc_cal, mod_freq=None, mod_dp_gain=1.0,
                 supply_ripple=None, mod_freq_dp=None, dp_cal=None):
        c = self.cfg
        # polartx: the direct point's own trajectory and the dp_cal hook
        mf = np.asarray(mod_freq, dtype=float) if mod_freq is not None else np.zeros(0)
        mf_dp = np.asarray(mod_freq_dp, dtype=float) if mod_freq_dp is not None else mf
        has_dpcal = dp_cal is not None
        if has_dpcal and mod_freq is None:
            raise ValueError("dp_cal correlates against mod_freq; pass mod_freq")
        if has_dpcal:
            dp_step, dp_st, dp_pr, dp_rec = _dp_cal_view(dp_cal, n_cycles)
            g_dp0 = float(dp_cal.value)          # the calibrator's start
        else:
            dp_step, dp_st, dp_pr, dp_rec = dp_cal_off, np.zeros(1), np.zeros(1), np.zeros((0, 0))
            g_dp0 = float(mod_dp_gain)
        dp_trace = np.empty(n_cycles if has_dpcal else 0)
        rng = np.random.default_rng(seed)
        tref = 1.0 / c.fref
        v_sup = supply_ripple_v(supply_ripple, n_cycles, tref)
        f_pull = pull_hz(c.osc, n_cycles, tref)
        fcw = c.fcw
        osc = Oscillator(c.osc, c.fref, rng, noise=noise, name="dco")
        tdc = TDC(c.tdc, rng, noise=noise)
        kdco_hat = c.osc.gain * (1.0 + c.kdco_est_error)
        otw_center = (c.fout - c.osc.f0) / c.osc.gain + f_start_offset / c.osc.gain
        cpp_nom = (1.0 / c.fout) / c.tdc.t_res    # nominal codes per period
        osc_noise = osc.noise_steps(n_cycles) if noise else np.zeros(n_cycles)
        jit_ref = self._ref_jitter(n_cycles, rng, noise)

        phase_err = np.empty(n_cycles)
        freq_out = np.empty(n_cycles)
        otw_rec = np.empty(n_cycles)
        # at most two draws per cycle (TDC jitter, the period-cal counter
        # noise), taken from a pool in the order the loop consumes them
        zn = rng.standard_normal(2 * n_cycles) if noise else np.zeros(0)
        has_tcal = tdc_cal is not None
        has_kcal = kdco_cal is not None
        tcal_trace = np.empty(n_cycles if has_tcal else 0)
        kcal_trace = np.empty(n_cycles if has_kcal else 0)
        kcal_trace[:] = kdco_cal.value if has_kcal else 0.0
        args: tuple[Any, ...] = (
            n_cycles, tref, float(c.fout), float(c.fref), float(fcw),
            *osc_law_args(c.osc), float(c.osc.pushing_hz_v), v_sup, f_pull, osc_noise,
            jit_ref, bool(noise), bool(calibration), zn,
            float(tdc.t_lsb_true), int(tdc.code_max), float(c.tdc.jitter_rms_s),
            *c.tdc.inl_scalars(), float(cpp_nom),
            has_tcal, tdc_cal.st if has_tcal else np.zeros(1),
            float(tdc_cal._ema) if has_tcal else 0.0, tcal_trace,
            has_kcal, kdco_cal.st if has_kcal else np.zeros(7),
            kdco_cal.halves if has_kcal else np.zeros(1),
            kdco_cal.ests if has_kcal else np.zeros(1),
            float(kdco_cal.amp) if has_kcal else 0.0,
            int(kdco_cal.meas_n) if has_kcal else 1,
            int(kdco_cal.rounds) if has_kcal else 0,
            int(kdco_cal.settle) if has_kcal else 0, kcal_trace,
            float(kdco_hat), float(otw_center), float(c.dlf.alpha), float(c.dlf.rho),
            np.asarray(c.dlf.iir_lambdas, dtype=float),
            np.zeros(len(c.dlf.iir_lambdas)), c.dco_dither_order > 0,
            mod_freq is not None, mf, g_dp0,
            mf_dp, has_dpcal, dp_step, dp_st, dp_pr, dp_rec, dp_trace,  # polartx
            phase_err, freq_out, otw_rec)
        adpll_tdc_kernel(*args)
        if has_dpcal and calibration:            # polartx
            _dp_cal_done(dp_cal, dp_st, dp_rec, dp_trace)
        if has_tcal and calibration:
            tdc_cal.trace.extend(tcal_trace.tolist())
        if has_kcal and calibration:
            # the object's trace grows only while its FCAL phase ran
            n_ran = min(n_cycles, max(kdco_cal.n, 0))
            kdco_cal.trace.extend(kcal_trace[:n_ran].tolist())

        t = np.arange(n_cycles) * tref
        lock = detect_lock(t, freq_out - c.fout, tol_hz=c.fout * 2e-5)
        sim = SimResult(fs=c.fref, f0=c.fout, t=t, phase_err_out=phase_err,
                        freq_out=freq_out, ctrl=otw_rec, lock_time_s=lock)
        if kdco_cal is not None:
            sim.cal_traces["kdco_hat"] = np.asarray(kdco_cal.trace)
        if tdc_cal is not None:
            sim.cal_traces["tdc_cpp"] = np.asarray(tdc_cal.trace)
        if has_dpcal:                            # polartx
            sim.cal_traces["dp_gain"] = dp_trace
        # TDC mode takes a fractional FCW natively (no MASH, no DTC), so the
        # fractional beat lands in the spectrum just as it does everywhere
        # else -- tabulate it instead of leaving the user to find it by eye
        frac = c.fcw % 1.0
        offs = frac_spur_offsets(frac, c.fref) if frac > 1e-9 else None
        offs = add_pull_offset(offs, c.osc, c.fref)
        if supply_ripple is not None and supply_ripple[1] < 0.45 * c.fref:
            offs = (offs or []) + [supply_ripple[1]]
        return postprocess(sim, int_band=c.int_band, spur_offsets=offs,
                           flicker_corner_hz=flicker_corner_hz(c))

    def _sim_bbpd(self, n_cycles, noise, calibration, seed, f_start_offset,
                  dtc_gain_init_error, dtc_gain_drift=None,
                  supply_ripple=None):
        c = self.cfg
        rng = np.random.default_rng(seed)
        tref = 1.0 / c.fref
        v_sup = supply_ripple_v(supply_ripple, n_cycles, tref)
        f_pull = pull_hz(c.osc, n_cycles, tref)
        osc = Oscillator(c.osc, c.fref, rng, noise=noise, name="dco")
        bb = BBPD(c.bb_jitter_rms_s, rng, noise=noise,
                  meta_window_s=c.bb_meta_window_s)
        frac = c.frac
        assert frac is not None      # __post_init__ requires it in this mode
        mash = frac.make_mash()
        frac_word = frac.frac_word
        n_int = int(c.fout // c.fref)
        otw_center = (c.fout - c.osc.f0) / c.osc.gain + f_start_offset / c.osc.gain

        dtc = None
        dtc_cal = None
        if frac.dtc is not None:
            from ..blocks.dtc import DTC
            dtc = DTC(frac.dtc, rng, noise=noise, gain_error=dtc_gain_init_error)
            if calibration:
                dtc_cal = frac.dtc_cal

        osc_noise = osc.noise_steps(n_cycles) if noise else np.zeros(n_cycles)
        jit_ref = self._ref_jitter(n_cycles, rng, noise)
        jit_div = self._div_jitter(n_cycles, rng, noise)

        phase_err = np.empty(n_cycles)
        freq_out = np.empty(n_cycles)
        otw_rec = np.empty(n_cycles)
        cal_trace = np.empty(n_cycles if dtc_cal is not None else 0)
        # at most three draws per cycle (DTC jitter, BBPD jitter, a
        # metastable coin), from a pool in the order the loop consumes them
        zn = rng.standard_normal(3 * n_cycles) if noise else np.zeros(0)
        dtc_args = dtc_kernel_args(frac.dtc)
        cal_args = cal_kernel_args(dtc_cal)
        cal_st = cal_args[2]
        args: tuple[Any, ...] = (
            n_cycles, tref, float(c.fout), int(n_int), *osc_law_args(c.osc),
            float(c.osc.pushing_hz_v), v_sup, f_pull, osc_noise, jit_ref, jit_div,
            bool(noise), zn, int(frac.mash_order), int(frac.bits), mash.st,
            int(frac_word), dtc is not None, *dtc_args,
            float(dtc_gain_init_error), dtc_gain_drift is not None,
            (np.asarray(dtc_gain_drift, dtype=float) if dtc_gain_drift is not None
             else np.zeros(0)),
            *cal_args, cal_trace,
            float(c.bb_jitter_rms_s), float(c.bb_meta_window_s), float(otw_center),
            float(c.dlf.alpha), float(c.dlf.rho), c.dco_dither_order > 0,
            phase_err, freq_out, otw_rec)
        n_meta, _ = adpll_bbpd_kernel(*args)
        if dtc_cal is not None:
            dtc_cal.load_state(cal_st)
            dtc_cal.trace.extend(cal_trace.tolist())
        bb.n_meta = n_meta

        t = np.arange(n_cycles) * tref
        lock = detect_lock(t, freq_out - c.fout, tol_hz=c.fout * 2e-5)
        sim = SimResult(fs=c.fref, f0=c.fout, t=t, phase_err_out=phase_err,
                        freq_out=freq_out, ctrl=otw_rec, lock_time_s=lock)
        if dtc_cal is not None:
            sim.cal_traces["dtc_gain"] = cal_trace
        if c.bb_meta_window_s > 0:
            frac_meta = bb.n_meta / max(n_cycles, 1)
            sim.extra["bbpd_metastable_frac"] = frac_meta
            sim.notes.append(
                f"BBPD decision was a coin flip on {frac_meta * 100:.1f}% of "
                "cycles (metastability window): that is lost detector gain, "
                "not added noise — the comparator still emits +/-1")
        offs = add_pull_offset(
            frac_spur_offsets(c.frac.frac, c.fref, fmin=8.0 * c.fref / n_cycles),
            c.osc, c.fref)
        return postprocess(sim, int_band=c.int_band, spur_offsets=offs,
                           flicker_corner_hz=flicker_corner_hz(c))
