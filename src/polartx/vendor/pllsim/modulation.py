# Vendored from pll_simulator@d7be4712: src/pllsim/modulation.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Two-point PLL modulation: GMSK trajectories and EVM.

A PLL transmits phase/frequency modulation through two summing points:
the LOWPASS point (FCW / DTC target — the loop tracks it, band-limited by
the loop BW) and the HIGHPASS point (direct DCO/VCO frequency push — the
loop fights it below the loop BW).  With matched gains the two responses
sum to exactly 1 at all offsets and the data rate can exceed the loop BW
by orders of magnitude; a direct-point gain error epsilon leaves a
residual error trajectory eps*phi_mod highpass-filtered by the loop.

Everything here runs on the reference-cycle grid of the time-domain
engines (one sample per fref cycle), so mod arrays plug straight into
simulate(..., mod_freq=...).
"""
from __future__ import annotations

import numpy as np

TWOPI = 2.0 * np.pi


def gmsk_trajectory(bits: np.ndarray, fref: float, rb: float,
                    bt: float = 0.5, h: float = 0.5,
                    n_total: int | None = None,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """GMSK frequency and ideal phase trajectories on the fref grid.

    bits: 0/1 array; rb: bit rate [b/s]; bt: Gaussian BT product;
    h: modulation index (0.5 for GMSK: +/- pi/2 per symbol).
    Returns (freq_dev_hz[n], phase_ideal_rad[n]) with n on the fref grid;
    zero-padded to n_total if given.  fref/rb (samples per symbol) need
    not be an integer.
    """
    bits = np.asarray(bits).astype(int)
    nrz = 2 * bits - 1
    sps = fref / rb
    n_sym = bits.size
    n = int(np.ceil(n_sym * sps))
    t_idx = np.arange(n)
    # NRZ waveform on the fref grid (symbol k spans [k*sps, (k+1)*sps))
    sym_idx = np.minimum((t_idx / sps).astype(int), n_sym - 1)
    m = nrz[sym_idx].astype(float)
    # Gaussian filter: sigma_t = sqrt(ln2)/(2*pi*B), B = bt*rb
    sigma_n = np.sqrt(np.log(2.0)) / (TWOPI * bt * rb) * fref
    span = int(np.ceil(4 * sigma_n))
    k = np.arange(-span, span + 1)
    taps = np.exp(-0.5 * (k / sigma_n) ** 2)
    taps /= taps.sum()
    m = np.convolve(m, taps, mode="same")
    freq = m * h * rb / 2.0                    # peak deviation h*Rb/2
    if n_total is not None:
        pad = np.zeros(n_total)
        pad[: min(n, n_total)] = freq[: min(n, n_total)]
        freq = pad
    phase = TWOPI * np.cumsum(freq) / fref
    return freq, phase


def evm(phase_actual: np.ndarray, phase_ideal: np.ndarray,
        skip: int = 0, detrend: bool = True, align: bool = True) -> dict:
    """Constant-envelope EVM from the phase-error trajectory.

    EVM_rms = rms(|exp(j*phi_a) - exp(j*phi_i)|) = rms(2*sin(dphi/2)).
    detrend removes the static phase offset and any residual frequency
    offset; align searches a few integer sample lags — a real demodulator
    does both (carrier and timing recovery).  The engines' phase record is
    one cycle behind the injected trajectory, so lag 1 is typical.
    """
    pa = np.asarray(phase_actual[skip:], dtype=float)
    pi_ = np.asarray(phase_ideal[skip:], dtype=float)

    def _one(d):
        if detrend:
            x = np.arange(d.size, dtype=float)
            a, b = np.polyfit(x, d, 1)
            d = d - (a * x + b)
        ev = 2.0 * np.sin(0.5 * d)
        return float(np.sqrt(np.mean(ev**2))), d

    lags = range(-2, 3) if align else (0,)
    best = None
    for lag in lags:
        d = pa - np.roll(pi_, lag)
        m = max(4, abs(lag))
        rms, dd = _one(d[m:-m or None])
        if best is None or rms < best[0]:
            best = (rms, dd, lag)
    rms, d, lag = best
    return {"evm_rms": rms, "evm_pct": 100.0 * rms,
            "evm_db": float(20.0 * np.log10(max(rms, 1e-12))),
            "phase_err_rms_deg": float(np.degrees(np.std(d))),
            "lag": lag}


def prbs(n: int, seed: int = 1) -> np.ndarray:
    """PRBS-15 bit sequence (x^15 + x^14 + 1)."""
    state = seed & 0x7FFF or 1
    out = np.empty(n, dtype=int)
    for i in range(n):
        bit = ((state >> 14) ^ (state >> 13)) & 1
        state = ((state << 1) | bit) & 0x7FFF
        out[i] = bit
    return out
