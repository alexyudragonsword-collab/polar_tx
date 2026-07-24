"""Differential EVM (DEVM) for the EDR DPSK payloads.

Stylized BT test-spec measurement: matched SRRC receive filter (TX SRRC
x RX SRRC = raised cosine, ISI-free symbol instants), delay/gain
alignment against the ideal burst, then the differential error
e_k = z_k - z_{k-1} exp(j dphi_k) referenced to the rotated symbol —
RMS DEVM limits are 0.20 (pi/4-DQPSK) / 0.13 (8DPSK).
"""
from __future__ import annotations

import numpy as np

from ..vendor.padpd.data.align import align_delay
from ..waveforms.base import Waveform
from ..waveforms.edr import _srrc


def devm(y: np.ndarray, wf: Waveform) -> dict:
    m = wf.meta
    sps, span = m["sps"], m["span"]
    h = _srrc(sps, m["rolloff"], span)
    h = h / np.sum(h * h)                    # unit RC peak after matching
    r = np.convolve(y, h, mode="full")[h.size // 2: h.size // 2 + y.size]
    ref = np.convolve(wf.x, h, mode="full")[h.size // 2: h.size // 2 + y.size]
    ref_a, r_a, info = align_delay(ref, r, max_lag=4 * sps)

    z = r_a[::sps] / info["gain"]
    n_sym = z.size
    dphi = m["dphi_ideal"][1:n_sym]
    w = z[:-1] * np.exp(1j * dphi)           # expected next symbol
    e = z[1:] - w
    s = slice(span, -span if span else None)
    rms = float(np.sqrt(np.mean(np.abs(e[s]) ** 2) /
                        np.mean(np.abs(w[s]) ** 2)))
    return {"devm_rms": rms, "devm_pct": 100.0 * rms,
            "devm_db": float(20.0 * np.log10(max(rms, 1e-12))),
            "symbols_rx": z, "lag_total": info["lag_total"]}
