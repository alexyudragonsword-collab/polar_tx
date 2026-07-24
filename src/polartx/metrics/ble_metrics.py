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


def acp_transient_db(y: np.ndarray, fs: float, *, offset_hz: float = 2e6,
                     ch_bw: float = 1e6) -> float:
    """Burst-transient adjacent-channel power, max-hold [dBc].

    The ramp specs (BT/GSM 'power-vs-time in the adjacent channel') are
    max-hold, not averaged: a keying step splatters for microseconds and
    a Welch average dilutes it below visibility.  Downconvert the
    adjacent channel, brick-wall to ch_bw, take the peak instantaneous
    power relative to the in-channel mean."""
    y = np.asarray(y, dtype=complex)
    n = y.size
    t = np.arange(n) / fs

    def _band_env(fc):
        z = y * np.exp(-2j * np.pi * fc * t)
        spec = np.fft.fft(z)
        f = np.fft.fftfreq(n, 1.0 / fs)
        spec[np.abs(f) > ch_bw / 2] = 0.0
        return np.abs(np.fft.ifft(spec)) ** 2

    p_in = _band_env(0.0).mean()
    p_adj = max(_band_env(offset_hz).max(), _band_env(-offset_hz).max())
    return float(10 * np.log10(p_adj / p_in))


def bt_acp(y: np.ndarray, fs: float, offsets_hz=(2e6, 3e6),
           ch_bw: float = 1e6, nfft: int = 8192) -> dict:
    """Bluetooth adjacent-channel power (stylized BR/EDR in-band spec):
    power in ch_bw-wide channels at +/-offsets relative to the in-channel
    power [dBc].  The BT limits are absolute dBm; relative to a 0-10 dBm
    TX the -20/-40 dBm class limits map to ~-30/-50 dBc."""
    from scipy import signal as sig
    f, pxx = sig.welch(y, fs=fs, nperseg=min(nfft, len(y)),
                       return_onesided=False, detrend=False)
    order = np.argsort(f)
    f, pxx = f[order], pxx[order]

    def _band(fc):
        m = (f >= fc - ch_bw / 2) & (f < fc + ch_bw / 2)
        return float(pxx[m].sum())

    p0 = _band(0.0)
    out = {}
    for off in offsets_hz:
        for sgn, tag in ((1, "+"), (-1, "-")):
            out[f"acp{tag}{off / 1e6:g}MHz_dbc"] = \
                float(10 * np.log10(_band(sgn * off) / p0))
    return out


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

    # certification-grade extensions --------------------------------
    # delta-f2-max proper: the RF-PHY suite takes the MAXIMUM deviation
    # within each symbol (not the center sample) and requires >= 99.9%
    # of symbols above 185 kHz (LE 1M)
    n_used = n_sym - 2 * discard_syms
    k0 = int(discard_syms * sps)
    per_sym_max = np.empty(n_used)
    for i in range(n_used):
        s0 = k0 + int(i * sps)
        seg = f_inst[s0: s0 + int(sps)]
        per_sym_max[i] = np.max(seg * sym[i])      # toward correct dir
    out["df_sym_max_hz"] = per_sym_max
    out["df2max_p001_hz"] = float(np.percentile(per_sym_max, 0.1))
    out["frac_above_185k"] = float(np.mean(per_sym_max >= 185e3))

    # carrier drift over the burst (spec: |drift| <= 50 kHz, rate <=
    # 400 Hz/us): slope of the symbol-center deviations
    k = np.arange(dev.size, dtype=float)
    slope = np.polyfit(k, dev, 1)[0]               # Hz per symbol
    out["drift_hz_per_us"] = float(slope * rate / 1e6)
    out["drift_hz_total"] = float(slope * dev.size)
    if wf.meta.get("pattern") == "11110000":
        # middle two bits of each 4-run, BLE delta-f1-avg convention
        k = np.arange(signed.size)
        mid = (k % 4 == 1) | (k % 4 == 2)
        out["df1_avg_hz"] = float(np.mean(np.abs(dev[mid])))
    if wf.meta.get("pattern") == "10101010":
        out["df2_avg_hz"] = float(np.mean(np.abs(dev)))
        out["df2_min_hz"] = float(np.min(signed))
    return out
