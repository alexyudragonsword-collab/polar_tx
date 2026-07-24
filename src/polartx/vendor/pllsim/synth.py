# Vendored from pll_simulator@d7be4712: src/pllsim/synth.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Loop synthesis: component values from bandwidth/phase-margin targets.

All synthesis is numeric against the library's own loop models (the same
grid-evaluated responses analyze() uses), so what you ask for is what
loop_metrics() reports afterwards — including the C3 loading of the R2 node
that breaks the textbook closed forms.

  design_cp_filter   : passive 2nd/3rd-order filter for any charge-pump-class
                       loop.  kdet [A/rad] composes the PD+CP gain:
                         CPPLL:  kdet = Icp/(2π·N)
                         SSPLL:  kdet = A·gm·τ_pulse·fref      (output-referred)
                         SPLL:   kdet = A·gm·τ_pulse·fref / N  (ref-referred)
  design_adpll_dlf   : PI coefficients (alpha, rho) against the exact z-domain
                       ADPLL loop including the z^-1 update delay
  sweep_bandwidth    : jitter-vs-UGB sweep over any pll factory
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .blocks.loopfilter import FilterDesign, LoopFilter
from .core.freqresp import FreqResponse, default_grid, loop_metrics
from .core.jitter import integrate_pn

TWOPI = 2.0 * np.pi


def cppll_kdet(icp: float, n_div: float) -> float:
    """Composite PD+CP gain [A/rad] of a PFD/CP loop."""
    return icp / (TWOPI * n_div)


def sspll_kdet(amp_v: float, gm: float, pulse_width: float, fref: float) -> float:
    """Composite PD gain [A/rad of OUTPUT phase] of a sub-sampling loop."""
    return amp_v * gm * pulse_width * fref


def spll_kdet(amp_v: float, gm: float, pulse_width: float, fref: float,
              n_div: float) -> float:
    """Composite PD gain [A/rad] of a reference-sampling loop (÷N to output)."""
    return amp_v * gm * pulse_width * fref / n_div


def _cp_gol(f: np.ndarray, filt: FilterDesign, kdet: float, kvco: float,
            tstep: float) -> FreqResponse:
    lf = LoopFilter(filt, tstep)
    z = FreqResponse(f, lf.transimpedance(f))
    return z * FreqResponse.integrator(f, TWOPI * kvco) * kdet


def design_cp_filter(kdet: float, kvco_hz_v: float, f_ugb: float, pm_deg: float,
                     fref: float, order: int = 3,
                     third_pole_ratio: float = 8.0) -> FilterDesign:
    """Synthesize a passive CP filter hitting (f_ugb, pm_deg) numerically.

    Free variables: total capacitance (loop-gain scale) and zero time
    constant (phase boost).  Parasitic poles are budgeted at
    third_pole_ratio × f_ugb; the solver works on the real network response,
    so C3 loading is included.  Raises if the target is infeasible.
    """
    if f_ugb > fref / 8:
        raise ValueError(f"f_ugb {f_ugb:.3g} Hz > fref/8: continuous-time CP "
                         "synthesis is not meaningful; lower the target")
    f = default_grid(max(f_ugb / 300, 10.0), min(300 * f_ugb, 2e9), 80)
    t3 = 1.0 / (TWOPI * third_pole_ratio * f_ugb)

    def build(log_ctot: float, log_t2: float, log_c2f: float) -> FilterDesign | None:
        ctot = float(np.exp(log_ctot))
        t2 = float(np.exp(log_t2))
        c2f = float(np.exp(log_c2f))          # C2 fraction of Ctot (free dof)
        if c2f > 0.4:
            return None
        if order == 3:
            c2 = c2f * ctot
            c3 = 0.5 * c2
            c1 = ctot - c2 - c3
        else:
            c2, c3 = c2f * ctot, 0.0
            c1 = ctot - c2
        if c1 <= 0:
            return None
        r2 = t2 / c1
        if order == 3:
            return FilterDesign(c1=c1, r2=r2, c2=c2, r3=t3 / c3, c3=c3)
        return FilterDesign(c1=c1, r2=r2, c2=c2)

    def residuals(p):
        filt = build(*p)
        if filt is None:
            return [10.0, 10.0, 10.0]
        m = loop_metrics(_cp_gol(f, filt, kdet, kvco_hz_v, 1.0 / fref))
        if not (np.isfinite(m.f_ugb) and np.isfinite(m.pm_deg)):
            return [10.0, 10.0, 10.0]
        # weak prior keeps the C2 fraction near a practical 5% when the two
        # hard targets leave it under-determined
        return [np.log(m.f_ugb / f_ugb), (m.pm_deg - pm_deg) / 30.0,
                0.05 * (p[2] - np.log(0.05))]

    # textbook 2nd-order initial guess
    phi = np.radians(min(pm_deg + np.degrees(np.arctan(1.0 / third_pole_ratio)), 85.0))
    wc = TWOPI * f_ugb
    t1 = max(abs((1.0 / np.cos(phi) - np.tan(phi)) / wc), 1e-3 / wc)
    t2_0 = 1.0 / (wc * wc * t1)
    ctot0 = kdet * TWOPI * kvco_hz_v / wc**2 * np.sqrt(
        (1 + (wc * t2_0) ** 2) / (1 + (wc * t1) ** 2))
    sol = least_squares(residuals, [np.log(ctot0), np.log(t2_0), np.log(0.05)],
                        xtol=1e-13, ftol=1e-13)
    filt = build(*sol.x)
    m = loop_metrics(_cp_gol(f, filt, kdet, kvco_hz_v, 1.0 / fref))
    if abs(np.log(m.f_ugb / f_ugb)) > 0.05 or abs(m.pm_deg - pm_deg) > 3.0:
        raise RuntimeError(
            f"CP filter synthesis failed: got {m.f_ugb / 1e6:.3f} MHz / "
            f"{m.pm_deg:.1f} deg for target {f_ugb / 1e6:.3f} MHz / "
            f"{pm_deg:.0f} deg — infeasible with this pole budget")
    return filt


def design_sspll_filter(k_q: float, kvco_hz_v: float, f_ugb: float,
                        pm_deg: float, fref: float,
                        third_pole_ratio: float = 8.0) -> FilterDesign:
    """Like design_cp_filter but solved on the EXACT DISCRETE sub-sampling
    loop (charge impulse + z-accumulator), which the CT approximation
    under-predicts by ~30% at the aggressive UGB/fref ratios SSPLLs use.

    k_q: PD charge gain [C/rad of output phase] = A·gm·τ_pulse.
    """
    ct = design_cp_filter(k_q * fref, kvco_hz_v, f_ugb, pm_deg, fref,
                          third_pole_ratio=third_pole_ratio)
    f = default_grid(max(f_ugb / 300, 10.0), 0.49 * fref, 100)
    zinv_arr = np.exp(-2j * np.pi * f / fref)

    def gol_z(filt: FilterDesign) -> FreqResponse:
        lf = LoopFilter(filt, 1.0 / fref)
        hq = FreqResponse(f, lf.charge_tf_z(f))
        acc = FreqResponse(f, zinv_arr / (1.0 - zinv_arr))
        return hq * acc * (k_q * TWOPI * kvco_hz_v / fref)

    def scale(p) -> FilterDesign:
        # rescale all caps (gain) and R2 (zero) from the CT solution
        kc, kr = np.exp(p[0]), np.exp(p[1])
        return FilterDesign(c1=ct.c1 * kc, r2=ct.r2 * kr, c2=ct.c2 * kc,
                            r3=ct.r3 * kr if ct.c3 else 0.0,
                            c3=ct.c3 * kc if ct.c3 else 0.0)

    def residuals(p):
        m = loop_metrics(gol_z(scale(p)))
        if not (np.isfinite(m.f_ugb) and np.isfinite(m.pm_deg)):
            return [10.0, 10.0]
        return [np.log(m.f_ugb / f_ugb), (m.pm_deg - pm_deg) / 30.0]

    sol = least_squares(residuals, [0.0, 0.0], xtol=1e-13, ftol=1e-13)
    filt = scale(sol.x)
    m = loop_metrics(gol_z(filt))
    if abs(np.log(m.f_ugb / f_ugb)) > 0.05 or abs(m.pm_deg - pm_deg) > 3.0:
        raise RuntimeError(
            f"SSPLL filter synthesis failed: {m.f_ugb / 1e6:.3f} MHz / "
            f"{m.pm_deg:.1f} deg for target {f_ugb / 1e6:.3f} MHz / {pm_deg:.0f} deg")
    return filt


# ------------------------------------------------------------------ ADPLL
def _adpll_gol(f: np.ndarray, alpha: float, rho: float, fref: float,
               iir_lambdas: tuple) -> FreqResponse:
    zinv = np.exp(-2j * np.pi * f / fref)
    l_z = alpha + rho / (1.0 - zinv)
    for lam in iir_lambdas:
        l_z = l_z * lam / (1.0 - (1.0 - lam) * zinv)
    return FreqResponse(f, l_z * zinv / (1.0 - zinv))


def design_adpll_dlf(fref: float, f_ugb: float, pm_deg: float,
                     iir_lambdas: tuple = ()) -> tuple[float, float]:
    """(alpha, rho) hitting (f_ugb, pm_deg) on the exact z-domain loop
    (normalized DCO gain, one-cycle update delay included)."""
    if f_ugb > fref / 6:
        raise ValueError("f_ugb > fref/6: no PI gain set can be stable there")
    f = default_grid(max(f_ugb / 300, 10.0), 0.49 * fref, 100)

    def residuals(p):
        alpha, rho = np.exp(p[0]), np.exp(p[1])
        m = loop_metrics(_adpll_gol(f, alpha, rho, fref, iir_lambdas))
        if not (np.isfinite(m.f_ugb) and np.isfinite(m.pm_deg)):
            return [10.0, 10.0]
        return [np.log(m.f_ugb / f_ugb), (m.pm_deg - pm_deg) / 30.0]

    alpha0 = TWOPI * f_ugb / fref
    rho0 = alpha0 * (TWOPI * f_ugb / fref) / 4.0
    sol = least_squares(residuals, [np.log(alpha0), np.log(rho0)],
                        xtol=1e-13, ftol=1e-13)
    alpha, rho = float(np.exp(sol.x[0])), float(np.exp(sol.x[1]))
    m = loop_metrics(_adpll_gol(f, alpha, rho, fref, iir_lambdas))
    if abs(np.log(m.f_ugb / f_ugb)) > 0.05 or abs(m.pm_deg - pm_deg) > 3.0:
        raise RuntimeError(
            f"DLF synthesis failed: got {m.f_ugb / 1e6:.3f} MHz / {m.pm_deg:.1f} deg "
            f"(target {f_ugb / 1e6:.3f} MHz / {pm_deg:.0f} deg)")
    return alpha, rho


# ------------------------------------------------------------------ sweeps
def sweep_bandwidth(make_pll, f_ugb_list, int_band=None):
    """Analyze a pll factory across UGB targets.

    make_pll(f_ugb) -> architecture instance (raise/skip on infeasible).
    Returns dict with 'f_ugb', 'jitter_fs', 'pm_deg', and per-source
    integrated-jitter breakdown 'sources' {name: [fs...]}.
    """
    out = {"f_ugb": [], "jitter_fs": [], "pm_deg": [], "sources": {}}
    for bw in f_ugb_list:
        try:
            pll = make_pll(bw)
            ar = pll.analyze()
        except (RuntimeError, ValueError):
            continue
        band = int_band or ar.int_band
        out["f_ugb"].append(ar.loop.f_ugb)
        out["jitter_fs"].append(ar.jitter_fs)
        out["pm_deg"].append(ar.loop.pm_deg)
        for name, s in ar.pn_breakdown.items():
            if name == "total":
                continue
            p = integrate_pn(ar.f, s, *band)
            fs = 1e15 * np.sqrt(p) / (TWOPI * ar.f0)
            out["sources"].setdefault(name, []).append(fs)
    for k in ("f_ugb", "jitter_fs", "pm_deg"):
        out[k] = np.asarray(out[k])
    return out
