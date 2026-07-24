"""Bluetooth LE GFSK waveforms (LE 1M / LE 2M, stylized: raw PDU bits only).

GFSK with BT = 0.5 and modulation index h = 0.5 (peak deviation
+/-rate/4 . h = +/-250 kHz @ 1 Msym/s, +/-500 kHz @ 2 Msym/s), built on the
Gaussian frequency-pulse engine adapted from pllsim.modulation.  The
returned Waveform carries the ideal frequency/phase trajectories so the
BLE metrics (phase EVM, delta-f1/delta-f2) need no demodulator model.
"""
from __future__ import annotations

import numpy as np

from ..vendor.pllsim.modulation import gmsk_trajectory, prbs
from .base import Waveform

TWOPI = 2.0 * np.pi

#: test payloads from the BLE RF-PHY test spec (repeated to length)
PATTERNS = {
    "prbs": None,          # PRBS-15
    "11110000": np.array([1, 1, 1, 1, 0, 0, 0, 0]),   # delta-f1 payload
    "10101010": np.array([1, 0, 1, 0, 1, 0, 1, 0]),   # delta-f2 payload
}


def ble_bits(n_bits: int, pattern: str = "prbs", seed: int = 1) -> np.ndarray:
    if pattern == "prbs":
        return prbs(n_bits, seed=seed)
    base = PATTERNS[pattern]
    reps = int(np.ceil(n_bits / base.size))
    return np.tile(base, reps)[:n_bits]


def gfsk_ble(n_bits: int, fs: float, rate: float = 1e6, *,
             pattern: str = "prbs", seed: int = 1, bt: float = 0.5,
             mod_index: float = 0.5) -> Waveform:
    """BLE GFSK burst on the fs grid.

    fs must give >= 8 samples/symbol for trustworthy EVM (engine-floor
    convention of the pllsim two-point examples); presets use fs = fref.
    """
    if fs / rate < 8:
        raise ValueError(f"fs/rate = {fs / rate:.1f} < 8 samples/symbol")
    bits = ble_bits(n_bits, pattern, seed)
    freq, phase = gmsk_trajectory(bits, fs, rate, bt=bt, h=mod_index)
    x = np.exp(1j * phase)
    return Waveform(x=x, fs=fs, bw=rate, kind="gfsk",
                    freq_ideal=freq, phase_ideal=phase,
                    meta={"bits": bits, "rate": rate, "pattern": pattern,
                          "bt": bt, "mod_index": mod_index})
