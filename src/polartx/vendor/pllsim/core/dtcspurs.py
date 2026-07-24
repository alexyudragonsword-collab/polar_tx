# Vendored from pll_simulator@d7be4712: src/pllsim/core/dtcspurs.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Deterministic fractional-spur prediction from the DTC code path.

Fractional spurs in DTC-cancelled fractional-N loops are not random: the
MASH residual sequence is periodic in the accumulator, so every systematic
error in the DTC transfer — code quantization, INL, residual gain error —
produces coherent tones at the fractional beat frequencies

    nu_k = (k * frac) mod 1,   f_spur = fold(nu_k) * fref .

This module replays the bit-true modulator through the DTC code mapping
(nominal gain assumed calibrated), projects the resulting timing-error
sequence onto the exact tone frequencies with a Hann-windowed DTFT (no FFT
bin leakage), and refers each tone to the output through the same transfer
the loop applies to DTC noise:

    spur [dBc] = 20 log10( 2*pi*fout*|A_k| * |NTF(f_spur)| / 2 )

The prediction is *mechanistic*: with a perfect DTC it reduces to the pure
quantization spurs; adding inl_sin/inl_poly or a gain residual reproduces
the measured worst-channel behavior (near-integer channels are worst — the
beat falls inside the loop bandwidth where |NTF| ~ 1).
"""
from __future__ import annotations

import numpy as np

TWOPI = 2.0 * np.pi


def dtc_error_sequence(frac_cfg, t_target_of, n_seq: int,
                       gain_eps: float = 0.0) -> np.ndarray:
    """Systematic DTC timing error [s] per reference cycle, bit-true.

    t_target_of(residual_ui) must reproduce the architecture's bipolar DTC
    target exactly as passed to DTC.delay(); the code mapping below mirrors
    DTC.delay() with gain_corr = 1 (gain calibrated), no random jitter.
    """
    d = frac_cfg.dtc
    mash = frac_cfg.make_mash()
    word = frac_cfg.frac_word
    code_max = (1 << d.n_bits) - 1
    half = d.range_s / 2.0

    codes = np.empty(n_seq, dtype=np.int64)
    t_req = np.empty(n_seq)
    if frac_cfg.mash_order == 1:
        # EFM1 residual is a pure accumulator ramp: vectorize
        acc = (np.arange(n_seq, dtype=np.int64) * word) % (1 << frac_cfg.bits)
        res = -acc.astype(float) / (1 << frac_cfg.bits)
        for i in range(n_seq):
            t_req[i] = t_target_of(res[i]) + half
    else:
        for i in range(n_seq):
            t_req[i] = t_target_of(mash.residual_ui()) + half
            mash.step(word)
    codes = np.clip(np.round(t_req / d.t_res).astype(np.int64), 0, code_max)

    err = codes * d.t_res - t_req
    if gain_eps:
        err = err + gain_eps * codes * d.t_res
    x = codes / code_max
    if d.inl_poly:
        err = err + np.polyval(d.inl_poly[::-1], x)
    if d.inl_sin:
        amp, cyc, ph = d.inl_sin
        err = err + amp * np.sin(TWOPI * cyc * x + ph)
    return err - err.mean()


def dtc_spur_table(frac_cfg, t_target_of, fref: float, fout: float,
                   ntf=None, kmax: int = 6, n_seq: int = 1 << 15,
                   fmin: float = 1e3, gain_eps: float = 0.0,
                   floor_dbc: float = -160.0) -> dict[float, float]:
    """Predicted fractional spurs {offset_hz: dbc} (SSB, relative to carrier).

    ntf: FreqResponse of the loop transfer applied to DTC phase error
    (h / h_lp in the architectures); None means |NTF| = 1.
    """
    err = dtc_error_sequence(frac_cfg, t_target_of, n_seq, gain_eps)
    n = np.arange(n_seq)
    w = 0.5 - 0.5 * np.cos(TWOPI * n / n_seq)          # Hann
    wsum = w.sum()

    out: dict[float, float] = {}
    for k in range(1, kmax + 1):
        nu = (k * frac_cfg.frac) % 1.0
        f_off = min(nu, 1.0 - nu) * fref
        if not (fmin < f_off < 0.45 * fref):
            continue
        key = round(f_off, 3)
        if key in out:
            continue
        a = 2.0 * np.abs(np.sum(w * err * np.exp(-1j * TWOPI * nu * n))) / wsum
        dphi = TWOPI * fout * a                         # peak phase deviation
        mag = 1.0
        if ntf is not None:
            mag = float(np.interp(np.log10(f_off), np.log10(ntf.f),
                                  np.abs(ntf.h)))
        dbc = 20.0 * np.log10(max(dphi * mag / 2.0, 1e-30))
        if dbc > floor_dbc:
            out[key] = dbc
    return out
