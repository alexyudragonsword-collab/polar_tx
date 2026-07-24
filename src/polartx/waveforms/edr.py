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
#: EDGE: 8PSK with a continuous 3pi/8 rotation per symbol
_3PI8_8PSK = _8DPSK + 3.0 * np.pi / 8.0


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
             rate_sym: float = 1e6, rolloff: float = 0.4, span: int = 16,
             seed: int = 1) -> Waveform:
    """Differential-PSK payload burst on the fs grid.

    Modes: "pi4dqpsk"/"8dpsk" (BT EDR2/EDR3, 1 Msym/s) and "3pi8_8psk"
    (EDGE: 8PSK with a 3pi/8 per-symbol rotation, 270.833 ksym/s — pass
    rate_sym=13e6/48).  fs must be an integer multiple of rate_sym.  The
    Waveform carries the ideal baseband, the symbol-instant indices and
    the ideal differential increments for the DEVM metric.
    """
    rs = rate_sym
    sps_f = fs / rs
    sps = int(round(sps_f))
    if abs(sps_f - sps) > 1e-6 or sps < 8:
        raise ValueError("fs must be an integer multiple of rate_sym, "
                         ">= 8 samples/symbol")
    table = {"8dpsk": _8DPSK, "pi4dqpsk": _PI4DQPSK,
             "3pi8_8psk": _3PI8_8PSK}[mode]
    bps = 2 if mode == "pi4dqpsk" else 3
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


def edr_packet(n_payload_syms: int, fs: float, *, mode: str = "8dpsk",
               n_header_bits: int = 126, guard_us: float = 5.0,
               seed: int = 1) -> Waveform:
    """Full EDR packet timing: GFSK access-code/header (1 Mb/s) ->
    guard (unmodulated carrier, ~5 us) -> DPSK sync+payload.

    The GFSK section is constant-envelope; the guard carries the
    GFSK-to-DPSK transition; the payload is SRRC DPSK.  meta['segments']
    holds (start, stop) sample indices for {'gfsk','guard','dpsk'} so
    segment metrics (header delta-f, payload DEVM) can be sliced from a
    chain output that is sample-aligned with the input.
    """
    from ..vendor.pllsim.modulation import gmsk_trajectory

    bits = prbs(n_header_bits, seed=seed + 100)
    _, ph_g = gmsk_trajectory(bits, fs, 1e6, bt=0.5, h=0.5)
    x_gfsk = np.exp(1j * ph_g)

    n_guard = int(round(guard_us * 1e-6 * fs))
    x_guard = np.full(n_guard, np.exp(1j * ph_g[-1]))   # phase-continuous

    wf_p = edr_dpsk(n_payload_syms, fs, mode=mode, seed=seed)
    x_dpsk = wf_p.x * np.exp(1j * np.angle(x_guard[-1]))

    x = np.concatenate([x_gfsk, x_guard, x_dpsk])
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    n0, n1 = x_gfsk.size, x_gfsk.size + n_guard
    meta = dict(wf_p.meta)
    meta.update({"packet": True, "header_bits": bits,
                 "segments": {"gfsk": (0, n0), "guard": (n0, n1),
                              "dpsk": (n1, x.size)},
                 "payload_meta": wf_p.meta, "payload_x": x_dpsk})
    return Waveform(x=x, fs=fs, bw=wf_p.bw, kind="dpsk", meta=meta)


def edge_waveform(n_syms: int, fs: float = 26e6, *, seed: int = 1,
                  rolloff: float = 0.3) -> Waveform:
    """EDGE 3pi/8-8PSK burst (stylized: SRRC in place of the linearized
    GMSK pulse) at 270.833 ksym/s; fs = 26 MHz gives 96 samples/symbol
    on the classic GSM crystal grid."""
    return edr_dpsk(n_syms, fs, mode="3pi8_8psk", rate_sym=13e6 / 48,
                    rolloff=rolloff, seed=seed)
