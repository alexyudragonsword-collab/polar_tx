# Vendored from pll_simulator@5564c64: src/pllsim/core/tdcspurs.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Deterministic fractional-spur prediction from TDC nonlinearity.

The TDC-based ADPLL has no DTC code sequence to replay, which is why its
analyze() used to hand back an empty spur table and a note saying to read
the number out of simulate().  The mechanism is nonetheless just as
deterministic as the DTC case, and it is worth being able to see before
running anything.

With a fractional FCW the TDC input sweeps one oscillator period at the
fractional beat rate: the position inside that period is

    u[n] = (n * frac) mod 1 ,

so any systematic INL is a fixed function g(u) sampled along that ramp.
Expand g in a Fourier series on [0, 1),

    g(u) = sum_k c_k exp(j 2 pi k u) ,

and each harmonic maps to a tone at fold(k*frac)*fref -- one spur per k,
with amplitude |c_k| in seconds at the PD input.  Referring that to the
output through the loop's own response,

    spur [dBc] = 20 log10( 2*pi*fout*|c_k| * |NTF(f_spur)| / 2 ) ,

the same accounting core.dtcspurs uses.  Nothing here is fitted: the
coefficients come from the declared INL shape, and integer k*frac values
(the near-integer channels) land the spur inside the loop bandwidth where
|NTF| ~ 1, which is exactly where the measurement is worst.

The one thing that is easy to get wrong is *which* variable the INL is a
function of.  TDCConfig declares the shape over the converter's code range,
but the loop only ever walks one oscillator period of it -- and a TDC is
deliberately built with range > Tosc, so that is a fraction

    span = Tosc / (code_max * t_lsb)  <  1

of the declared axis.  A shape with a whole number of cycles across the range
therefore presents a *fractional* number of cycles to the loop, which is
discontinuous at the wrap and so spreads over every k instead of leaving a
single line.  Ignoring the span overstates the worst spur (4.3 dB for the
100 MHz ADPLL preset) and misses the harmonics entirely.
"""
from __future__ import annotations

import numpy as np

TWOPI = 2.0 * np.pi


def code_span(tdc, fout: float) -> float:
    """Fraction of the TDC's code range one oscillator period occupies.

    Above 1.0 the converter cannot cover a period and saturates every cycle,
    which is a configuration error rather than a spur mechanism; the caller
    is expected to say so.
    """
    code_max = (1 << tdc.n_bits) - 1
    t_lsb = tdc.t_res * (1.0 + getattr(tdc, "gain_error", 0.0))
    return (1.0 / fout) / (code_max * t_lsb)


def inl_fourier_coeffs(inl_sin, span: float = 1.0, n_harm: int = 16,
                       n_grid: int = 4096) -> np.ndarray:
    """|c_k| for k = 1..n_harm of the INL along the ramp the loop walks.

    inl_sin is (amplitude_s, cycles, phase) exactly as TDCConfig declares it,
    i.e. over the full code range.  `span` maps that axis onto the one
    oscillator period the input actually sweeps (see the module docstring);
    span = 1.0 recovers the plain decomposition over the declared range.

    A whole number of *effective* cycles (cyc*span) gives one nonzero
    coefficient; anything else spreads across k, which is the honest answer
    and not an artefact -- it is what the wrap discontinuity really radiates.
    """
    amp, cyc, ph = inl_sin
    u = np.arange(n_grid) / n_grid
    # mirror TDC.measure's saturation: past the end of the range the code
    # clamps, so the INL stops moving rather than continuing round the circle
    x = np.minimum(u * span, 1.0)
    g = amp * np.sin(TWOPI * cyc * x + ph)
    ck = np.fft.rfft(g) / n_grid
    out = np.zeros(n_harm)
    for k in range(1, n_harm + 1):
        if k < ck.size:
            out[k - 1] = 2.0 * abs(ck[k])      # one-sided amplitude
    return out


def tdc_inl_spur_table(tdc, frac: float, fref: float, fout: float,
                       ntf=None, n_harm: int = 16, f_min: float = 1e3,
                       dyn_db: float = 40.0) -> dict[float, float]:
    """{offset_hz: spur_dbc} from a sinusoidal TDC INL on a fractional channel.

    `tdc` is the TDCConfig -- the resolution and bit count are needed as well
    as the INL shape, because they set how much of that shape the loop sees.

    Only harmonics within `dyn_db` of the worst spur are returned.  The span
    correction spreads a single declared cycle over every k, and a table that
    listed all sixteen down to -110 dBc would bury the two or three lines that
    are actually findings under a dozen that no one will ever measure.

    Returns an empty table for an integer channel: with frac = 0 the TDC sits
    on one code and its INL is a static offset, not a tone.
    """
    inl_sin = tdc.inl_sin
    if not inl_sin or frac <= 0.0 or frac >= 1.0:
        return {}
    amps = inl_fourier_coeffs(inl_sin, code_span(tdc, fout), n_harm)
    # a whole number of INL cycles leaves one real coefficient and a floor of
    # FFT round-off; publishing those as -350 dBc entries would read as
    # findings rather than as zeros
    floor = 1e-6 * float(np.max(amps)) if np.max(amps) > 0 else 0.0
    out: dict[float, float] = {}
    for k, a in enumerate(amps, start=1):
        if a <= floor:
            continue
        nu = (k * frac) % 1.0
        f_sp = min(nu, 1.0 - nu) * fref
        if not (f_min < f_sp < 0.5 * fref):
            continue
        gain = 1.0
        if ntf is not None:
            gain = float(np.interp(f_sp, ntf.f, np.abs(ntf.h)))
        phi1 = TWOPI * fout * a * gain          # peak output phase [rad]
        dbc = float(20.0 * np.log10(max(phi1 / 2.0, 1e-30)))
        key = round(f_sp, 3)
        # two harmonics can fold onto the same offset; powers add
        if key in out:
            dbc = float(10.0 * np.log10(10 ** (dbc / 10) + 10 ** (out[key] / 10)))
        out[key] = dbc
    if out and dyn_db > 0:
        keep = max(out.values()) - dyn_db
        out = {k: v for k, v in out.items() if v >= keep}
    return out
