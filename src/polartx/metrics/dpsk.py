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


def packet_metrics(y: np.ndarray, wf: Waveform) -> dict:
    """Segment metrics for an edr_packet burst: header GFSK frequency
    deviation + payload DEVM, each on its own sample slice (the chain
    output is sample-aligned with the input)."""
    from .ble_metrics import freq_deviation
    seg = wf.meta["segments"]
    g0, g1 = seg["gfsk"]
    d0, d1 = seg["dpsk"]
    hdr_wf = Waveform(x=wf.x[g0:g1], fs=wf.fs, bw=1e6, kind="gfsk",
                      meta={"bits": wf.meta["header_bits"], "rate": 1e6,
                            "pattern": "prbs"})
    hdr = freq_deviation(y[g0:g1], hdr_wf)
    pay_wf = Waveform(x=wf.meta["payload_x"], fs=wf.fs, bw=wf.bw,
                      kind="dpsk", meta=wf.meta["payload_meta"])
    pay = devm(y[d0:d1], pay_wf)
    return {"header_dev_avg_hz": hdr["dev_avg_hz"],
            "header_wrong_sign_frac": hdr["wrong_sign_frac"],
            "payload_devm_pct": pay["devm_pct"],
            "payload": pay, "header": hdr}
