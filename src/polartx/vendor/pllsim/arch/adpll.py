# Vendored from pll_simulator@d7be4712: src/pllsim/arch/adpll.py
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

from dataclasses import dataclass, field

import numpy as np

from ..blocks.oscillator import OscConfig, Oscillator
from ..blocks.tdc import BBPD, TDC, TDCConfig
from ..core.colored import synth_from_psd
from ..core.deltasigma import Mash11
from ..core.engine import detect_lock, postprocess
from ..core.freqresp import FreqResponse, default_grid, loop_metrics
from ..core.jitter import ipn_dbc, rms_jitter_fs
from ..core.noise import (DcoQuantPhase, FlickerFloorPhase, NoisePath,
                          NoiseSource, ShapedQuantization, output_psd)
from ..core.results import AnalysisResult, SimResult
from .base import PLLBase
from .frac import FracConfig, frac_spur_offsets  # polartx: extracted from cppll.py

TWOPI = 2.0 * np.pi


@dataclass
class DLFConfig:
    """Digital loop filter: L(z) = alpha + rho/(1-z^-1), then IIR stages
    lam/(1-(1-lam) z^-1) each."""

    alpha: float
    rho: float
    iir_lambdas: tuple = ()


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
    def analyze(self, f: np.ndarray | None = None) -> AnalysisResult:
        c = self.cfg
        if f is None:
            f = default_grid(1e2, 1e9)
        notes = []
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
        else:
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
                eps = getattr(c.frac.dtc, "gain_error_residual", 0.01)
                paths.append(NoisePath(
                    ShapedQuantization(name="dsm_residual", unit="rad^2/Hz",
                                       q=TWOPI * eps, fs=c.fref,
                                       order=c.frac.mash_order - 1), h))

        m = loop_metrics(gol)
        bd = output_psd(paths, f)
        jit = rms_jitter_fs(f, bd["total"], c.fout, *c.int_band)
        if m.f_ugb > c.fref / 10:
            notes.append("UGB > fref/10: discrete loop peaking significant")
        spurs = {}
        if c.mode == "dtc_bbpd" and c.frac.dtc is not None:
            from ..core.dtcspurs import dtc_spur_table
            eps = getattr(c.frac.dtc, "gain_error_residual", 0.01)
            for off, dbc in dtc_spur_table(
                    c.frac, lambda r: r / c.fout, c.fref, c.fout,
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
        sigma_t = max(c.bb_jitter_rms_s, 1e-15)
        gol = None
        for _ in range(8):
            kbb = np.sqrt(2.0 / np.pi) / sigma_t                  # 1/s
            # e = Kbb * dt_pd ; dt_pd = phi_out/(2π fout)
            det = kbb / (TWOPI * c.fout)
            gol = self._dlf_fr(f) * self._dco_phase_fr(f) * det
            h = gol.feedback()
            err = 1.0 - h
            # output jitter contributions at the PD
            s_dco = c.osc.leeson("dco").psd(f) * err.mag2()
            from ..core.jitter import integrate_pn
            var_phi = integrate_pn(f, s_dco, 1e4, c.fref / 2)
            var_t = var_phi / (TWOPI * c.fout) ** 2 \
                + q_dtc**2 / 12.0 + c.bb_jitter_rms_s**2
            sigma_new = np.sqrt(var_t)
            if abs(sigma_new - sigma_t) < 1e-18:
                break
            sigma_t = 0.5 * sigma_t + 0.5 * sigma_new
        kbb = np.sqrt(2.0 / np.pi) / sigma_t
        return gol, kbb, sigma_t

    # ------------------------------------------------------------ simulate
    def simulate(self, n_cycles: int, *, noise: bool = True, calibration: bool = True,
                 seed: int = 0, f_start_offset: float = 0.0,
                 kdco_cal=None, tdc_cal=None, dtc_gain_init_error: float = 0.0,
                 mod_freq: np.ndarray | None = None, mod_dp_gain: float = 1.0,
                 dtc_gain_drift: np.ndarray | None = None,
                 mod_freq_dp: np.ndarray | None = None,
                 dp_cal=None) -> SimResult:
        """mod_freq: two-point modulation frequency trajectory [Hz] on the
        fref grid (see pllsim.modulation); injected at the FCW (lowpass)
        and DCO (highpass) points.  mod_dp_gain scales the direct point:
        1.0 = matched, 1+eps models a direct-path gain error.  TDC mode.
        mod_freq_dp (polartx extension): separate trajectory for the
        direct point — models a range-limited direct-modulation DAC
        (FCW path keeps the full trajectory).  None = use mod_freq.
        dp_cal (polartx extension): background two-point gain calibrator
        (SignSignLMS-style .step(err, regressor)/.value): its value
        REPLACES mod_dp_gain each cycle and is updated from the phase
        error against the modulation regressor; trace recorded in
        cal_traces['dp_gain']."""
        if self.cfg.mode == "tdc":
            return self._sim_tdc(n_cycles, noise, calibration, seed,
                                 f_start_offset, kdco_cal, tdc_cal,
                                 mod_freq, mod_dp_gain, mod_freq_dp,
                                 dp_cal)
        return self._sim_bbpd(n_cycles, noise, calibration, seed,
                              f_start_offset, dtc_gain_init_error,
                              dtc_gain_drift)

    def _ref_jitter(self, n_cycles, rng, noise):
        c = self.cfg
        if not noise:
            return np.zeros(n_cycles)
        src = FlickerFloorPhase.from_spot("ref", c.ref_pn_dbchz, c.ref_pn_fc)
        return synth_from_psd(src.psd, c.fref, n_cycles, rng) / (TWOPI * c.fref)

    def _sim_tdc(self, n_cycles, noise, calibration, seed, f_start_offset,
                 kdco_cal, tdc_cal, mod_freq=None, mod_dp_gain=1.0,
                 mod_freq_dp=None, dp_cal=None):
        if mod_freq_dp is None:
            mod_freq_dp = mod_freq
        dp_trace = np.empty(n_cycles) if dp_cal is not None else None
        c = self.cfg
        rng = np.random.default_rng(seed)
        tref = 1.0 / c.fref
        fcw = c.fcw
        osc = Oscillator(c.osc, c.fref, rng, noise=noise, name="dco")
        tdc = TDC(c.tdc, rng, noise=noise)
        kdco_hat = c.osc.gain * (1.0 + c.kdco_est_error)
        otw_center = (c.fout - c.osc.f0) / c.osc.gain + f_start_offset / c.osc.gain

        # DLF state
        acc = 0.0
        iir_state = [0.0] * len(c.dlf.iir_lambdas)
        qerr = 0.0                       # 1st-order DCO dither state
        phi_v = 0.0                      # DCO phase [UI]
        r_acc = 0.0                      # reference accumulator [UI]
        cpp_nom = (1.0 / c.fout) / c.tdc.t_res    # nominal codes per period

        osc_noise = osc.noise_steps(n_cycles) if noise else np.zeros(n_cycles)
        prev_phi_n = 0.0
        jit_ref = self._ref_jitter(n_cycles, rng, noise)

        phase_err = np.empty(n_cycles)
        freq_out = np.empty(n_cycles)
        otw_rec = np.empty(n_cycles)
        fv = c.osc.f0 + c.osc.gain * otw_center

        for n in range(n_cycles):
            d_phi_n = (osc_noise[n] - prev_phi_n) / TWOPI     # UI
            prev_phi_n = osc_noise[n]
            phi_v += fv * tref + d_phi_n
            r_acc += fcw
            if mod_freq is not None:
                # lowpass point [UI] — one cycle behind the direct point:
                # the fv used for phi_v this cycle carries mod_freq[n-1]
                r_acc += (mod_freq[n - 1] if n > 0 else 0.0) * tref
            # reference jitter shifts the sampling instant of the DCO phase
            phi_sample = phi_v + jit_ref[n] * fv
            count = np.floor(phi_sample)
            frac_ui = phi_sample - count
            tdco = 1.0 / fv
            code = tdc.measure(frac_ui * tdco)
            if tdc_cal is not None and calibration:
                cpp_meas = tdco / tdc.t_lsb_true + rng.normal(0.0, 0.5) if noise \
                    else tdco / tdc.t_lsb_true
                tdc_cal.step(cpp_meas)
                tdc_ui = tdc_cal.code_to_ui(code)
            else:
                tdc_ui = code / cpp_nom
            e_ui = r_acc - (count + tdc_ui)

            if kdco_cal is not None and calibration and not kdco_cal.done:
                # open-loop FCAL phase: loop frozen, OTW stepped +/-A; the
                # frequency is measured from the counter (phase slope)
                kdco_cal.step(fv)
                kdco_hat = kdco_cal.value
                otw = otw_center + kdco_cal.perturbation
                r_acc = count + tdc_ui       # keep PD aligned for loop closure
                acc = 0.0
            else:
                # DLF
                acc += c.dlf.rho * e_ui
                x = c.dlf.alpha * e_ui + acc
                for i, lam in enumerate(c.dlf.iir_lambdas):
                    iir_state[i] += lam * (x - iir_state[i])
                    x = iir_state[i]
                otw = x * c.fref / kdco_hat + otw_center
            if dp_cal is not None and calibration:
                dp_cal.step(e_ui, mod_freq[n])
                dp_trace[n] = dp_cal.value
            if mod_freq_dp is not None:              # highpass (direct) point
                g_dp = dp_cal.value if dp_cal is not None else mod_dp_gain
                otw += mod_freq_dp[n] * g_dp / kdco_hat
            # DCO quantization with optional 1st-order dither
            if c.dco_dither_order > 0:
                otw_q = np.floor(otw + qerr)
                qerr = otw + qerr - otw_q
            else:
                otw_q = np.round(otw)
            fv = c.osc.f0 + c.osc.gain * otw_q

            phase_err[n] = TWOPI * (phi_v - (n + 1) * fcw)
            freq_out[n] = fv
            otw_rec[n] = otw_q

        t = np.arange(n_cycles) * tref
        lock = detect_lock(t, freq_out - c.fout, tol_hz=c.fout * 2e-5)
        sim = SimResult(fs=c.fref, f0=c.fout, t=t, phase_err_out=phase_err,
                        freq_out=freq_out, ctrl=otw_rec, lock_time_s=lock)
        if kdco_cal is not None:
            sim.cal_traces["kdco_hat"] = np.asarray(kdco_cal.trace)
        if tdc_cal is not None:
            sim.cal_traces["tdc_cpp"] = np.asarray(tdc_cal.trace)
        if dp_cal is not None:
            sim.cal_traces["dp_gain"] = dp_trace
        return postprocess(sim, int_band=c.int_band)

    def _sim_bbpd(self, n_cycles, noise, calibration, seed, f_start_offset,
                  dtc_gain_init_error, dtc_gain_drift=None):
        c = self.cfg
        rng = np.random.default_rng(seed)
        tref = 1.0 / c.fref
        osc = Oscillator(c.osc, c.fref, rng, noise=noise, name="dco")
        bb = BBPD(c.bb_jitter_rms_s, rng, noise=noise)
        mash = c.frac.make_mash()
        frac_word = c.frac.frac_word
        n_int = int(c.fout // c.fref)
        otw_center = (c.fout - c.osc.f0) / c.osc.gain + f_start_offset / c.osc.gain

        dtc = None
        dtc_cal = None
        if c.frac.dtc is not None:
            from ..blocks.dtc import DTC
            dtc = DTC(c.frac.dtc, rng, noise=noise, gain_error=dtc_gain_init_error)
            if calibration:
                dtc_cal = c.frac.dtc_cal

        acc = 0.0
        qerr = 0.0
        phi_out = 0.0
        t_div = 0.0
        n_next = n_int
        osc_noise = osc.noise_steps(n_cycles) if noise else np.zeros(n_cycles)
        prev_phi_n = 0.0
        jit_ref = self._ref_jitter(n_cycles, rng, noise)

        phase_err = np.empty(n_cycles)
        freq_out = np.empty(n_cycles)
        otw_rec = np.empty(n_cycles)
        cal_trace = np.empty(n_cycles) if dtc_cal is not None else None
        fv = c.osc.f0 + c.osc.gain * otw_center

        for n in range(n_cycles):
            t_ref = n * tref + jit_ref[n]
            residual_ui = mash.residual_ui()
            if dtc is not None:
                if dtc_gain_drift is not None:
                    dtc.gain_error = dtc_gain_drift[n]
                t_ref += dtc.delay(residual_ui / c.fout)
            e = bb.sample(t_div - t_ref)

            if dtc_cal is not None:
                dtc_cal.step(e, residual_ui)
                dtc.gain_corr = dtc_cal.value
                cal_trace[n] = dtc_cal.value

            acc += c.dlf.rho * e
            otw = c.dlf.alpha * e + acc + otw_center
            if c.dco_dither_order > 0:
                otw_q = np.floor(otw + qerr)
                qerr = otw + qerr - otw_q
            else:
                otw_q = np.round(otw)
            fv = c.osc.f0 + c.osc.gain * otw_q

            n_next = n_int + mash.step(frac_word)
            d_osc = osc_noise[n] - prev_phi_n
            prev_phi_n = osc_noise[n]
            t_div += n_next / fv - d_osc / (TWOPI * fv)

            phi_out += TWOPI * (fv - c.fout) * tref + d_osc
            phase_err[n] = phi_out
            freq_out[n] = fv
            otw_rec[n] = otw_q

        t = np.arange(n_cycles) * tref
        lock = detect_lock(t, freq_out - c.fout, tol_hz=c.fout * 2e-5)
        sim = SimResult(fs=c.fref, f0=c.fout, t=t, phase_err_out=phase_err,
                        freq_out=freq_out, ctrl=otw_rec, lock_time_s=lock)
        if dtc_cal is not None:
            sim.cal_traces["dtc_gain"] = cal_trace
        offs = frac_spur_offsets(c.frac.frac, c.fref, fmin=8.0 * c.fref / n_cycles)
        return postprocess(sim, int_band=c.int_band, spur_offsets=offs)
