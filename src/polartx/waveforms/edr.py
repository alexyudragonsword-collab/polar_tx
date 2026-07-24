"""Bluetooth EDR DPSK payload waveforms (stylized: payload symbols only).

EDR2 = pi/4-DQPSK (2 Mb/s), EDR3 = 8DPSK (3 Mb/s), both 1 Msym/s with
square-root-raised-cosine pulse shaping (roll-off 0.4).  Differential
encoding: the data lives in the phase CHANGE between consecutive symbol
instants, so the receiver metric is DEVM, not absolute EVM.  Unlike the
GFSK packets these are NOT constant-envelope (PAPR ~2-3 dB) — through a
polar TX the envelope path and DPA are genuinely exercised.
"""
from __future__ import annotations

import numpy as np

from ..vendor.pllsim.modulation import prbs
from .base import Waveform

TWOPI = 2.0 * np.pi

#: differential phase increments (Gray-coded index -> radians)
_PI4DQPSK = np.array([1, 3, -1, -3]) * np.pi / 4.0
_8DPSK = np.array([0, 1, 3, 2, 7, 6, 4, 5]) * np.pi / 4.0


def _srrc(sps: int, rolloff: float, span: int) -> np.ndarray:
    """Square-root raised cosine taps, unit peak, `span` symbols long."""
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    b = rolloff
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = 1.0 - b + 4.0 * b / np.pi
        elif b > 0 and abs(abs(4.0 * b * ti) - 1.0) < 1e-9:
            h[i] = b / np.sqrt(2.0) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * b))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * b)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - b))
                    + 4 * b * ti * np.cos(np.pi * ti * (1 + b))) \
                / (np.pi * ti * (1 - (4 * b * ti) ** 2))
    return h / h.max()


def edr_dpsk(n_syms: int, fs: float, *, mode: str = "8dpsk",
             rolloff: float = 0.4, span: int = 16, seed: int = 1) -> Waveform:
    """EDR DPSK payload burst at symbol rate 1 Msym/s on the fs grid.

    fs must be an integer multiple of 1 MHz (samples/symbol).  The
    Waveform carries the ideal baseband, the symbol-instant indices and
    the ideal differential increments for the DEVM metric.
    """
    rs = 1e6
    sps_f = fs / rs
    sps = int(round(sps_f))
    if abs(sps_f - sps) > 1e-9 or sps < 8:
        raise ValueError("fs must be an integer multiple of 1 MHz, >= 8 sps")
    table = {"8dpsk": _8DPSK, "pi4dqpsk": _PI4DQPSK}[mode]
    bps = 3 if mode == "8dpsk" else 2
    bits = prbs(n_syms * bps, seed=seed).reshape(n_syms, bps)
    idx = bits @ (1 << np.arange(bps)[::-1])
    dphi = table[idx]
    phase_sym = np.cumsum(dphi)
    symbols = np.exp(1j * phase_sym)

    # SRRC pulse shaping on the fs grid
    up = np.zeros(n_syms * sps, dtype=complex)
    up[::sps] = symbols
    h = _srrc(sps, rolloff, span)
    x = np.convolve(up, h, mode="full")[h.size // 2: h.size // 2 + up.size]
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))

    return Waveform(x=x, fs=fs, bw=rs * (1 + rolloff), kind="dpsk",
                    meta={"mode": mode, "rate_sym": rs, "sps": sps,
                          "dphi_ideal": dphi, "symbols": symbols,
                          "rolloff": rolloff, "span": span, "seed": seed})
