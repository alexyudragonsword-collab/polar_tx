# Vendored from PA_DPD@08b9725: src/padpd/gain_modulation.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Power-gain-modulation identification: measure tau, don't guess it.

A PA with thermal / bias / trapping memory has an effective gain that
follows its recent envelope power. :class:`~padpd.pa.spline_state.
StateConditionedSpline` models that with slow IIR power states — but its
``state_alphas`` (per-sample smoothing factors, alpha = exp(-1/(tau*fs)))
must match the device's actual time constants, which are unknown for a
new PA. This module closes that loop with the classic step-response
experiment:

    1. short full-power burst   (lets stateful DUTs calibrate, then cool)
    2. long low-power settle    (reach the cold steady state)
    3. step up to full power    -> gain droop:
           dG(t) = Sum a_k (1 - e^{-t/tau_k})
    4. step back down           -> gain recovery:
           dG(t) = Sum b_k e^{-t/tau_k}

A constant-envelope drive isolates the state dynamics: within a segment
the static AM/AM contribution is constant, so any gain trajectory IS the
modulation. Heating and cooling are fitted independently (1..max_poles
exponentials, extra poles kept only when they clearly reduce the
residual); differing heating/cooling constants indicate trapping or
bias hysteresis rather than plain linear thermal RC behavior.

The result converts directly into a StateConditionedSpline recipe:

    res = identify_gain_modulation(pa, fs=320e6)
    model = StateConditionedSpline.from_signal(
        x, state_alphas=res.state_alphas(fs), ...)

Like the two-tone module, this sizes the *structure* from a cheap
characterization; the coefficients are still trained on measured data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _fit_multiexp(t: np.ndarray, g: np.ndarray, n_poles: int):
    """LS fit of complex g(t) = g_inf + sum_k c_k exp(-t/tau_k) with
    shared real taus and complex residues (magnitude droop and AM-PM
    drift are fitted jointly — on some PAs the modulation is almost
    purely phase). Returns (taus, |c_k| weights..., g_inf, rms)."""
    from scipy.optimize import curve_fit

    g = np.asarray(g, dtype=complex)
    target = np.concatenate([g.real, g.imag])

    def model(tt, *p):
        n = len(tt) // 2
        ts = tt[:n]
        out = np.full(n, p[0] + 1j * p[1], dtype=complex)
        for k in range(n_poles):
            c = p[2 + 3 * k] + 1j * p[3 + 3 * k]
            out = out + c * np.exp(-ts / p[4 + 3 * k])
        return np.concatenate([out.real, out.imag])

    span = complex(g[0] - g[-1])
    p0 = [float(g[-1].real), float(g[-1].imag)]
    for k in range(n_poles):
        p0 += [span.real / n_poles, span.imag / n_poles,
               float(t[-1]) / (3.0 ** (n_poles - k))]
    lo = [-np.inf, -np.inf] + [-np.inf, -np.inf, float(t[1])] * n_poles
    hi = [np.inf, np.inf] + [np.inf, np.inf, 50.0 * float(t[-1])] * n_poles
    tt = np.concatenate([t, t])
    popt, _ = curve_fit(model, tt, target, p0=p0, bounds=(lo, hi),
                        maxfev=20000)
    resid = float(np.sqrt(np.mean((target - model(tt, *popt)) ** 2)))
    taus = [float(popt[4 + 3 * k]) for k in range(n_poles)]
    cs = [abs(popt[2 + 3 * k] + 1j * popt[3 + 3 * k])
          for k in range(n_poles)]
    order = np.argsort(taus)
    return ([taus[i] for i in order], [cs[i] for i in order],
            complex(popt[0], popt[1]), resid)


def _bin_average(t: np.ndarray, g: np.ndarray, n_bins: int):
    """Time-bin the trajectory: gain modulation is slow, so averaging
    out sample-rate noise (and virtual-DUT state staircases) leaves the
    step response itself — without it the fit residual is noise-floor
    dominated and pole-count selection cannot discriminate."""
    if len(t) <= n_bins:
        return t, g
    n = len(t) - len(t) % n_bins
    return (t[:n].reshape(n_bins, -1).mean(axis=1),
            g[:n].reshape(n_bins, -1).mean(axis=1))


def _fit_step(t: np.ndarray, g: np.ndarray, max_poles: int = 2):
    """Fit with increasing pole count; keep an extra pole only when it
    cuts the residual by >10% (multi-exponential fits are ill-posed —
    an extra pole must earn its place, but thermal responses genuinely
    are multi-pole and an under-fitted single tau costs real dB in the
    downstream state model)."""
    best = None
    for n in range(1, max_poles + 1):
        try:
            fit = _fit_multiexp(t, g, n)
        except RuntimeError:
            continue
        if best is None or fit[3] < 0.9 * best[3]:
            best = fit
    if best is None:
        raise RuntimeError("step-response fit did not converge")
    return best


@dataclass
class GainModulationResult:
    """Identified power-gain-modulation dynamics of a PA."""

    fs: float
    significant: bool
    droop_db: float                       # |gain| change after step-up
    phase_drift_deg: float = 0.0          # AM-PM drift after step-up
    taus_heat_s: list = field(default_factory=list)
    weights_heat: list = field(default_factory=list)
    taus_cool_s: list = field(default_factory=list)
    weights_cool: list = field(default_factory=list)
    fit_rms: float = 0.0
    # time constants of the slowest pole that must fit inside the
    # observation before the heating/cooling asymmetry is trustworthy
    min_tau_spans: float = 4.0
    # raw trajectories for plotting / residual inspection
    t_s: np.ndarray | None = None
    gain_heat: np.ndarray | None = None
    gain_cool: np.ndarray | None = None

    @property
    def observation_s(self) -> float:
        """Duration of one step observation segment."""
        if self.t_s is None or len(self.t_s) < 2:
            return float("nan")
        return float(self.t_s[-1])

    @property
    def hysteresis_ratio(self) -> float:
        """Slowest cooling tau / slowest heating tau. ~1 for a linear
        thermal RC; far from 1 suggests trapping / bias hysteresis —
        but only when :attr:`hysteresis_reliable` holds."""
        if not self.taus_heat_s or not self.taus_cool_s:
            return float("nan")
        return self.taus_cool_s[-1] / self.taus_heat_s[-1]

    @property
    def hysteresis_reliable(self) -> bool:
        """Whether the observation window constrains the slowest pole.

        A multi-exponential fit cannot pin a time constant the capture
        barely spans: with an observation only a couple of tau long the
        slow residue and tau trade off almost freely, and the heating
        and cooling fits land on different points of that valley — a
        spurious hysteresis ratio on a perfectly linear thermal RC (a
        linear-RC DUT read 0.92 at 2.5 tau spans, against 0.999 at 8).
        Requires at least ``min_tau_spans`` time constants of both fits
        inside the window; ``state_alphas`` (from the heating fit) stays
        usable either way — it is this *asymmetry verdict* that needs
        the margin.

        The backed-off cooling segment excites the state less than the
        heating one, so it is the weaker fit; with a long window it is
        nevertheless accurate (8 tau spans: cooling 7.3/47.8 us against
        a truth of 8/50, ratio 0.999). Readings of 0.47 and 0.51 that
        earlier versions of this docstring cited as estimator bias came
        from a bug in ThermalReferencePA, fixed 2026-10-03, which put a
        block-periodic error into every capture; the backed-off segment
        is where it was largest relative to the signal.
        """
        obs = self.observation_s
        if not (self.taus_heat_s and self.taus_cool_s) or not obs > 0:
            return False
        slowest = max(self.taus_heat_s[-1], self.taus_cool_s[-1])
        return obs >= self.min_tau_spans * slowest

    def state_alphas(self, fs: float | None = None) -> tuple:
        """Per-sample IIR factors for StateConditionedSpline."""
        fs = self.fs if fs is None else fs
        return tuple(float(np.exp(-1.0 / (tau * fs)))
                     for tau in self.taus_heat_s)

    def rationale(self) -> str:
        if not self.significant:
            return (f"gain step response is flat ({self.droop_db:+.3f} dB,"
                    f" {self.phase_drift_deg:+.2f} deg) -> no power-gain "
                    "modulation; plain SMP/SplineGMP suffices")
        taus = ", ".join(f"{t*1e6:.1f}us" for t in self.taus_heat_s)
        bits = [f"gain drifts {self.droop_db:+.2f} dB / "
                f"{self.phase_drift_deg:+.1f} deg after a power step; "
                f"heating taus [{taus}] -> "
                f"state_alphas "
                f"{tuple(round(a, 6) for a in self.state_alphas())}"]
        r = self.hysteresis_ratio
        if not self.hysteresis_reliable:
            slowest = max(self.taus_heat_s[-1] if self.taus_heat_s else 0.0,
                          self.taus_cool_s[-1] if self.taus_cool_s else 0.0)
            bits.append(
                f"heating/cooling asymmetry NOT assessed (ratio {r:.2f} "
                f"unreliable): the observation is "
                f"{self.observation_s*1e6:.0f}us "
                f"but the slowest tau is {slowest*1e6:.1f}us — re-run the "
                f"probe with t_obs >= {self.min_tau_spans*slowest*1e6:.0f}us "
                "to judge trapping/bias hysteresis")
        elif np.isfinite(r) and not 0.5 <= r <= 2.0:
            bits.append(f"cooling {r:.1f}x slower than heating -> "
                        "trapping/bias hysteresis suspected (a linear "
                        "thermal RC cannot capture the asymmetry)")
        return "; ".join(bits)


def _result_from_trajectories(fs: float, t: np.ndarray,
                              g_heat: np.ndarray, g_cool: np.ndarray,
                              max_poles: int, droop_threshold_db: float,
                              phase_threshold_deg: float
                              ) -> GainModulationResult:
    """Shared fitting path: significance test, bin-average, multi-exp
    fit of the heating and cooling gain trajectories."""
    droop_db = float(20 * np.log10(np.abs(g_heat[-1])
                                   / np.abs(g_heat[0])))
    phase_deg = float(np.degrees(np.angle(g_heat[-1] / g_heat[0])))
    res = GainModulationResult(fs=float(fs), significant=False,
                               droop_db=droop_db,
                               phase_drift_deg=phase_deg, t_s=t,
                               gain_heat=g_heat, gain_cool=g_cool)
    if (abs(droop_db) < droop_threshold_db
            and abs(phase_deg) < phase_threshold_deg):
        return res

    tb, gb_heat = _bin_average(t, g_heat, n_bins=512)
    _, gb_cool = _bin_average(t, g_cool, n_bins=512)
    taus_h, bs_h, _, rms_h = _fit_step(tb, gb_heat, max_poles)
    taus_c, bs_c, _, _ = _fit_step(tb, gb_cool, max_poles)
    tot_h = sum(abs(b) for b in bs_h) or 1.0
    tot_c = sum(abs(b) for b in bs_c) or 1.0
    res.significant = True
    res.taus_heat_s = taus_h
    res.weights_heat = [float(abs(b) / tot_h) for b in bs_h]
    res.taus_cool_s = taus_c
    res.weights_cool = [float(abs(b) / tot_c) for b in bs_c]
    res.fit_rms = rms_h
    return res


def identify_gain_modulation_capture(x: np.ndarray, y: np.ndarray,
                                     fs: float, t_guard_s: float = 1e-6,
                                     max_poles: int = 2,
                                     droop_threshold_db: float = 0.05,
                                     phase_threshold_deg: float = 0.5,
                                     level_tol: float = 0.02
                                     ) -> GainModulationResult:
    """OFFLINE identification from a recorded step-probe capture.

    For measured data the experiment cannot drive the DUT interactively;
    instead the probe sequence is transmitted once and (x, y) recorded.
    ``x`` must be the constant-envelope probe actually sent — reference
    burst, low-power settle, step up, step down (the same sequence
    :func:`identify_gain_modulation` transmits). Segment boundaries are
    recovered from the |x| level changes, so exact durations don't
    matter: the LAST high-amplitude segment is the heating observation
    and the low segment after it the cooling one. Everything downstream
    (guard window, binning, multi-exponential fits, significance
    thresholds) is identical to the interactive path.
    """
    x = np.asarray(x, dtype=complex)
    y = np.asarray(y, dtype=complex)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length")
    lev = np.abs(x) / max(float(np.abs(x).max()), 1e-30)
    bounds = np.flatnonzero(np.abs(np.diff(lev)) > level_tol) + 1
    segments = [s for s in np.split(np.arange(len(x)), bounds)
                if len(s) >= 64]
    if len(segments) < 2:
        raise ValueError("capture does not contain a level step "
                         "(need constant-envelope segments)")
    heat, cool = segments[-2], segments[-1]
    if np.median(lev[heat]) <= np.median(lev[cool]):
        raise ValueError("last two segments are not a step-down pair "
                         "(expected ... high (heat) -> low (cool))")
    n_guard = min(int(round(t_guard_s * fs)), len(heat) // 4,
                  len(cool) // 4)
    g_heat = y[heat][n_guard:] / x[heat][n_guard:]
    g_cool = y[cool][n_guard:] / x[cool][n_guard:]
    n = min(len(g_heat), len(g_cool))
    t = np.arange(n) / fs
    return _result_from_trajectories(fs, t, g_heat[:n], g_cool[:n],
                                     max_poles, droop_threshold_db,
                                     phase_threshold_deg)


def step_probe_drive(fs: float, a_hi: float = 1.5,
                     a_lo_ratio: float = 0.35, t_obs_s: float = 2e-4,
                     settle_factor: float = 5.0) -> np.ndarray:
    """The step-probe drive sequence, for RECORDING on real hardware.

    Transmit this once, capture the PA output, and feed the pair to
    :func:`identify_gain_modulation_capture` (store both in the
    complete-source container's ``step`` group). Mirrors the sequence
    :func:`identify_gain_modulation` transmits interactively.
    """
    n_obs = max(int(round(t_obs_s * fs)), 64)
    n_settle = max(int(round(settle_factor * t_obs_s * fs)), n_obs)
    a_lo = a_hi * a_lo_ratio
    return np.concatenate([
        np.full(n_obs // 4, a_hi, dtype=complex),   # reference burst
        np.full(n_settle, a_lo, dtype=complex),     # cool to steady state
        np.full(n_obs, a_hi, dtype=complex),        # step up (heating)
        np.full(n_obs, a_lo, dtype=complex),        # step down (cooling)
    ])


def identify_gain_modulation(pa, fs: float, a_hi: float = 1.5,
                             a_lo_ratio: float = 0.35,
                             t_obs_s: float = 2e-4,
                             settle_factor: float = 5.0,
                             t_guard_s: float = 1e-6,
                             max_poles: int = 2,
                             droop_threshold_db: float = 0.05,
                             phase_threshold_deg: float = 0.5
                             ) -> GainModulationResult:
    """Run the step-response experiment against a PA callable.

    ``t_obs_s`` must exceed the slowest expected time constant (a
    truncated observation biases the fit fast). ``a_hi`` is the probe
    amplitude — it must sit in the PA's *compressed* upper amplitude
    range, where the state actually moves the gain (default 1.5 suits
    unit-RMS baseband, i.e. the upper occupancy of an OFDM waveform; a
    small-signal probe is nearly blind to gain modulation). The low
    level is ``a_hi * a_lo_ratio``. ``t_guard_s`` discards the start of
    each observation segment: an amplitude step also excites the PA's
    *electrical* memory (matching-network FIR transient, nanosecond
    scale), which is not gain modulation and would otherwise fake a
    droop on a perfectly static PA. Stateful DUTs (``reset()``) are
    cold-started first. Significance requires the step response to move
    either |gain| or phase beyond the thresholds.
    """
    if hasattr(pa, "reset"):
        pa.reset()
    n_obs = max(int(round(t_obs_s * fs)), 64)
    n_settle = max(int(round(settle_factor * t_obs_s * fs)), n_obs)
    n_guard = min(int(round(t_guard_s * fs)), n_obs // 4)
    a_lo = a_hi * a_lo_ratio

    pa(np.full(n_obs // 4, a_hi, dtype=complex))   # reference calibration
    pa(np.full(n_settle, a_lo, dtype=complex))     # cool to steady state
    y_heat = pa(np.full(n_obs, a_hi, dtype=complex))
    y_cool = pa(np.full(n_obs, a_lo, dtype=complex))

    t = np.arange(n_obs - n_guard) / fs
    g_heat = np.asarray(y_heat, dtype=complex)[n_guard:] / a_hi
    g_cool = np.asarray(y_cool, dtype=complex)[n_guard:] / a_lo
    return _result_from_trajectories(fs, t, g_heat, g_cool, max_poles,
                                     droop_threshold_db,
                                     phase_threshold_deg)
