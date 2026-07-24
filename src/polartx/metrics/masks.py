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


def default_mask(wf: Waveform):
    if wf.kind == "gfsk":
        return ble_mask(wf.meta.get("rate", 1e6))
    if wf.kind == "dpsk":
        return ble_mask(1e6)      # BR/EDR channels share the 1 MHz raster
    return default_wifi_mask(wf.bw)
