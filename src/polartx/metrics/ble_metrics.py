"""BLE modulation-quality metrics (stylized RF-PHY test-suite versions).

phase_evm: constant-envelope EVM from the phase-error trajectory (adapted
pllsim.modulation.evm — detrend + integer-lag alignment = carrier/timing
recovery).  freq_deviation: symbol-center frequency deviations for the
delta-f1 (11110000 payload) and delta-f2 (10101010 payload) measurements.
"""
from __future__ import annotations

import numpy as np

from ..vendor.pllsim.modulation import evm as _phase_evm
from ..waveforms.base import Waveform

TWOPI = 2.0 * np.pi


def phase_evm(y: np.ndarray, wf: Waveform, skip: int = 0) -> dict:
    """EVM of the transmitted phase trajectory vs the ideal GFSK phase."""
    return _phase_evm(np.unwrap(np.angle(y)), wf.phase_ideal, skip=skip)


def freq_deviation(y: np.ndarray, wf: Waveform, discard_syms: int = 4) -> dict:
    """Per-symbol center frequency deviation statistics.

    Demodulates f_inst = dphi/dt / 2pi, removes the static frequency
    offset, and samples symbol centers.  For the 11110000 payload the BLE
    suite averages |deviation| over the middle two bits of each 4-run
    (delta-f1-avg); for 10101010 every symbol counts and the headline
    number is the minimum |deviation| (delta-f2-max criterion: >= 99.9%
    of symbols above 185 kHz for LE 1M).
    """
    rate = wf.meta["rate"]
    bits = wf.meta["bits"]
    sps = wf.fs / rate
    phase = np.unwrap(np.angle(y))
    f_inst = np.gradient(phase) * wf.fs / TWOPI
    f_inst -= np.mean(f_inst)                     # carrier-offset removal
    n_sym = min(bits.size, int(phase.size / sps) - 1)
    centers = ((np.arange(n_sym) + 0.5) * sps).astype(int)
    dev = f_inst[centers][discard_syms:n_sym - discard_syms]
    sym = (2 * bits[discard_syms:n_sym - discard_syms].astype(int) - 1)
    signed = dev * sym                            # + = correct direction
    out = {"dev_hz": dev, "dev_avg_hz": float(np.mean(np.abs(dev))),
           "dev_min_hz": float(np.min(signed)),
           "wrong_sign_frac": float(np.mean(signed < 0))}
    if wf.meta.get("pattern") == "11110000":
        # middle two bits of each 4-run, BLE delta-f1-avg convention
        k = np.arange(signed.size)
        mid = (k % 4 == 1) | (k % 4 == 2)
        out["df1_avg_hz"] = float(np.mean(np.abs(dev[mid])))
    if wf.meta.get("pattern") == "10101010":
        out["df2_avg_hz"] = float(np.mean(np.abs(dev)))
        out["df2_min_hz"] = float(np.min(signed))
    return out
