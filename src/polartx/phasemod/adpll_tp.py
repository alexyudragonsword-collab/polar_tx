"""Narrowband phase modulator: ADPLL with two-point injection.

Two-point modulation sums the data at the FCW path (lowpass through the
loop, closed-loop response h) and the direct DCO push (highpass, 1 - h);
matched gains sum to exactly 1 at every offset, so the data rate may
exceed the loop bandwidth.  A direct-path gain error eps leaves
eps * (highpassed phase) as the error trajectory — the EVM-limiting spec.

Two engines, cross-checked by tests:

- mode="response" (default, full-frame fast): filters the commanded phase
  by the exact z-domain composite H_tp(f) = h + g_eff (1 - h) built from
  the vendored ADPLL loop model, then adds output phase noise synthesized
  from the analyze() budget.  Linearized: no TDC wrapping or dither x
  modulation coupling.
- mode="event" (validation/noise truth): resamples the frequency
  trajectory onto the fref grid and runs the vendored cycle-accurate
  ADPLL.simulate(mod_freq=...) engine.  Python-loop cost ~1-2 Mcycles/s;
  the comparison floor scales as 1/(fref/BW) samples — use >= 8 samples
  per symbol and validate on short segments (pllsim ex17 convention).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from ..vendor.pllsim.arch.adpll import ADPLL
from ..vendor.pllsim.core.colored import synth_from_psd
from .base import PhaseModResult, PhaseModulator

TWOPI = 2.0 * np.pi


class ADPLLTwoPoint(PhaseModulator):
    def __init__(self, pll: ADPLL, *, dp_gain: float = 1.0,
                 mode: str = "response", settle_cycles: int = 40_000,
                 dp_range_hz: float | None = None):
        """dp_range_hz: tuning range of the direct-modulation DAC (+/-).
        Constant-envelope trajectories need ~ the peak deviation, but
        polar phase paths slew hard at envelope nulls (a pi flip in one
        sample = fs/2 instantaneous deviation) — a finite range clips
        there and splatters the spectrum.  Modeled in event mode (the
        clip is nonlinear); both modes report the required range in
        diagnostics ['dp_required_range_hz']."""
        if mode not in ("response", "event"):
            raise ValueError("mode must be 'response' or 'event'")
        self.pll = pll
        self.dp_gain = dp_gain
        self.mode = mode
        self.settle_cycles = settle_cycles
        self.dp_range_hz = dp_range_hz
        self.dp_cal = None      # background two-point gain calibrator
                                # (SignSignLMS), event mode only
        self._ana = None

    # ----------------------------------------------------------- helpers
    @property
    def _g_eff(self) -> float:
        """Effective direct-path gain: dp DAC gain over the Kdco estimate
        error (the direct point is scaled by 1/kdco_hat but pushes the
        true Kdco)."""
        return self.dp_gain / (1.0 + self.pll.cfg.kdco_est_error)

    def analyze(self):
        if self._ana is None:
            self._ana = self.pll.analyze()
        return self._ana

    def h_tp(self, f: np.ndarray) -> np.ndarray:
        """Composite two-point phase transfer on offset grid f (f > 0)."""
        c = self.pll.cfg
        if c.mode == "tdc":
            gol, _ = self.pll._gol_tdc(f)
        else:
            gol, _, _ = self.pll._gol_bbpd(f)
        h = gol.feedback().h
        return h + self._g_eff * (1.0 - h)

    def _noise_psd_interp(self):
        ana = self.analyze()
        lf = np.log10(ana.f)
        ls = np.log10(np.maximum(ana.pn_breakdown["total"], 1e-30))

        def psd(f):
            return 10.0 ** np.interp(np.log10(np.maximum(f, ana.f[0])), lf, ls)

        return psd

    # ------------------------------------------------------------- modes
    def modulate(self, phase_cmd, fs_bb, *, noise=True, seed=0):
        phase_cmd = np.asarray(phase_cmd, dtype=float)
        if fs_bb > self.pll.cfg.fref:
            raise ValueError("fs_bb must be <= fref (z-domain validity)")
        if self.mode == "response":
            return self._mod_response(phase_cmd, fs_bb, noise, seed)
        return self._mod_event(phase_cmd, fs_bb, noise, seed)

    def _mod_response(self, phase_cmd, fs_bb, noise, seed):
        n = phase_cmd.size
        # remove the endpoint-connecting ramp so the FFT filtering is
        # circular-clean; H_tp(0) = 1 exactly, so the ramp passes as-is
        k = np.arange(n)
        ramp = phase_cmd[0] + (phase_cmd[-1] - phase_cmd[0]) * k / max(n - 1, 1)
        resid = phase_cmd - ramp
        spec = np.fft.rfft(resid)
        f = np.fft.rfftfreq(n, 1.0 / fs_bb)
        htp = np.ones(f.size, dtype=complex)
        htp[1:] = self.h_tp(f[1:])
        out = ramp + np.fft.irfft(spec * htp, n=n)
        f_dev = np.diff(phase_cmd) * fs_bb / TWOPI
        diag = {"mode": "response", "g_eff": self._g_eff,
                "dp_required_range_hz": float(np.abs(f_dev).max())}
        if self.dp_range_hz is not None and \
                diag["dp_required_range_hz"] > self.dp_range_hz:
            diag["dp_range_exceeded"] = True   # linear model cannot clip
        if noise:
            rng = np.random.default_rng(seed)
            pn = synth_from_psd(self._noise_psd_interp(), fs_bb, n, rng)
            out = out + pn
            diag["pn_jitter_fs"] = self.analyze().jitter_fs
        return PhaseModResult(phase_out=out, diagnostics=diag)

    def _mod_event(self, phase_cmd, fs_bb, noise, seed):
        c = self.pll.cfg
        up_f = c.fref / fs_bb
        up = int(round(up_f))
        if abs(up_f - up) > 1e-9:
            raise ValueError(f"fref/fs_bb = {up_f} must be an integer "
                             "for event mode")
        f_dev = np.diff(phase_cmd, prepend=phase_cmd[:1]) * fs_bb / TWOPI
        mod_ref = f_dev if up == 1 else resample_poly(f_dev, up, 1)
        n_mod = mod_ref.size
        settle = self.settle_cycles
        mod = np.zeros(settle + n_mod + 4)
        mod[settle:settle + n_mod] = mod_ref
        mod_dp = mod
        clip_frac = 0.0
        if self.dp_range_hz is not None:
            mod_dp = np.clip(mod, -self.dp_range_hz, self.dp_range_hz)
            clip_frac = float(np.mean(mod_dp[settle:] != mod[settle:]))
        if self.dp_cal is not None:
            self.dp_cal.value = self.dp_gain     # start from the estimate
        sim = self.pll.simulate(mod.size, noise=noise, seed=seed,
                                mod_freq=mod, mod_dp_gain=self.dp_gain,
                                mod_freq_dp=mod_dp, dp_cal=self.dp_cal)
        ideal = TWOPI * np.cumsum(mod) / c.fref
        actual = sim.phase_err_out

        # integer-lag align (record runs ~1 cycle behind the injection),
        # then remove static phase + residual frequency offset — what a
        # receiver's carrier recovery does
        w0 = settle + min(4000, n_mod // 4)
        w1 = settle + n_mod
        best = None
        for lag in range(-2, 3):
            d = np.roll(actual, -lag)[w0:w1] - ideal[w0:w1]
            kk = np.arange(d.size, dtype=float)
            a, b = np.polyfit(kk, d, 1)
            rms = float(np.std(d - (a * kk + b)))
            if best is None or rms < best[0]:
                best = (rms, lag)
        _, lag = best
        seg = np.roll(actual, -lag)[settle:settle + n_mod]
        d = seg - ideal[settle:settle + n_mod]
        kk = np.arange(d.size, dtype=float)
        a, b = np.polyfit(kk, d, 1)
        phase_ref = seg - (a * kk + b)
        out = phase_ref[::up][:phase_cmd.size]
        return PhaseModResult(
            phase_out=out,
            diagnostics={"mode": "event", "lag": lag, "sim": sim,
                         "residual_rms_rad": best[0],
                         "samples_per_ref_cycle": up,
                         "dp_required_range_hz": float(np.abs(mod).max()),
                         "dp_clip_frac": clip_frac})
