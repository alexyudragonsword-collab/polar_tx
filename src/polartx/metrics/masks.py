"""Spectral masks per application (stylized engineering templates, not
conformance clauses — same caveat as the vendored WiFi mask)."""
from __future__ import annotations

import numpy as np

from ..vendor.padpd.metrics import default_wifi_mask
from ..waveforms.base import Waveform


def ble_mask(rate: float = 1e6):
    """Bluetooth LE in-band emission template, relative to peak.

    Core-spec in-band limits measured in 100 kHz: <= -20 dBc at
    |f-fc| = 2 MHz and <= -30 dBc at >= 3 MHz (LE 1M; offsets scale
    with the 2 MHz channel raster for LE 2M).  Piecewise-linear
    (offset_hz, dBr) breakpoints, symmetric — vendored check_mask format.
    """
    s = rate / 1e6
    return np.array([
        (0.0e6, 0.0),
        (1.0e6 * s, 0.0),
        (2.0e6 * s, -20.0),
        (3.0e6 * s, -30.0),
        (10.0e6 * s, -30.0),
    ])


def lte_sem(bw: float = 20e6):
    """E-UTRA-style spectrum-emission template relative to peak PSD.

    Stylized engineering simplification of the 36.101 general SEM:
    flat in-channel, -25 dBr within the first MHz past the channel edge,
    -35 dBr out to edge+5 MHz, -45 dBr beyond (the absolute dBm/MHz
    limits mapped to a ~23 dBm-class TX)."""
    e = bw / 2
    return np.array([
        (0.0, 0.0),
        (e, 0.0),
        (e + 1e6, -25.0),
        (e + 5e6, -35.0),
        (e + 10e6, -45.0),
        (e + 30e6, -45.0),
    ])


def default_mask(wf: Waveform):
    if wf.kind == "gfsk":
        return ble_mask(wf.meta.get("rate", 1e6))
    if wf.kind == "dpsk":
        return ble_mask(1e6)      # BR/EDR channels share the 1 MHz raster
    if wf.kind == "ofdm" and wf.meta.get("scs_hz") == 15e3:
        return lte_sem(wf.bw)
    return default_wifi_mask(wf.bw)
