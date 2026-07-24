"""Generalized OFDM: the vendored padpd 802.11-engine with configurable
subcarrier spacing (adapted from PA_DPD src/padpd/waveform/ofdm.py).

GenOFDMConfig only re-derives the numerology properties; generation,
raised-cosine windowing and demodulation are the vendored functions
untouched, so the vendored EVM metric works on the result and the WiFi
presets are bit-exact against padpd (regression-tested).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..vendor.padpd.waveform.ofdm import (OFDMConfig, OFDMWaveform,
                                          demodulate_ofdm, generate_ofdm,
                                          papr_db)
from .base import Waveform

WIFI_SCS_HZ = 78.125e3
#: 802.11ax/be active-tone table (vendored values)
WIFI_ACTIVE_TONES = {20e6: 242, 40e6: 484, 80e6: 996,
                     160e6: 1992, 320e6: 3984}


@dataclass
class GenOFDMConfig(OFDMConfig):
    """OFDMConfig with free subcarrier spacing and active-tone count.

    scs_hz sets the numerology (78.125 kHz WiFi, 15 kHz LTE, 30/60/120 kHz
    5G NR); n_active_tones overrides the occupancy (default: WiFi table
    when it applies, else ~94% of the FFT, the vendored fallback).
    """

    scs_hz: float = WIFI_SCS_HZ
    n_active_tones: int | None = None
    fft_size_override: int | None = None    # decouple FFT grid from the
                                            # channel BW (LTE/NR style)

    @property
    def fft_size(self) -> int:  # type: ignore[override]
        if self.fft_size_override is not None:
            return self.fft_size_override
        n = self.bandwidth_hz / self.scs_hz
        if abs(n - round(n)) > 1e-9:
            raise ValueError("bandwidth must be a multiple of scs_hz "
                             "(or set fft_size_override)")
        return int(round(n))

    @property
    def sample_rate_hz(self) -> float:  # type: ignore[override]
        # the physical rate follows the FFT grid, not the channel BW
        # (identical to the vendored bw*oversampling when fft = bw/scs)
        return self.fft_size * self.scs_hz * self.oversampling

    @property
    def n_active(self) -> int:  # type: ignore[override]
        if self.n_active_tones is not None:
            return self.n_active_tones
        if self.scs_hz == WIFI_SCS_HZ and self.bandwidth_hz in WIFI_ACTIVE_TONES:
            return WIFI_ACTIVE_TONES[self.bandwidth_hz]
        return 2 * int(0.47 * self.fft_size)


def ofdm_waveform(cfg: GenOFDMConfig) -> Waveform:
    """Generate and wrap into the chain's Waveform container."""
    ref = generate_ofdm(cfg)
    return Waveform(x=ref.x, fs=cfg.sample_rate_hz, bw=cfg.bandwidth_hz,
                    kind="ofdm", ofdm_ref=ref,
                    meta={"qam": cfg.qam_order, "scs_hz": cfg.scs_hz,
                          "papr_db": papr_db(ref.x)})


def wifi_waveform(bw: float = 160e6, qam: int = 1024, *, n_symbols: int = 8,
                  oversampling: int = 4, seed: int = 0) -> Waveform:
    """802.11ax/be-style channel (bit-exact vs the vendored generator)."""
    return ofdm_waveform(GenOFDMConfig(
        bandwidth_hz=bw, qam_order=qam, n_symbols=n_symbols,
        oversampling=oversampling, seed=seed))


#: LTE channel BW -> (FFT size @ 15 kHz SCS, resource blocks)
LTE_NUMEROLOGY = {1.4e6: (128, 6), 3e6: (256, 15), 5e6: (512, 25),
                  10e6: (1024, 50), 15e6: (1536, 75), 20e6: (2048, 100)}


def lte_waveform(bw: float = 20e6, qam: int = 64, *, n_symbols: int = 28,
                 oversampling: int = 4, seed: int = 0) -> Waveform:
    """E-UTRA-style downlink-grid channel: 15 kHz SCS, standard FFT/RB
    table (20 MHz: 2048-FFT / 1200 tones, fs = 30.72 * os MS/s), normal-
    CP-like 144/2048 guard.  Stylized: data tones only, no DMRS/PSS."""
    if bw not in LTE_NUMEROLOGY:
        raise ValueError(f"bw must be one of {sorted(LTE_NUMEROLOGY)}")
    fft, n_rb = LTE_NUMEROLOGY[bw]
    return ofdm_waveform(GenOFDMConfig(
        bandwidth_hz=bw, qam_order=qam, n_symbols=n_symbols,
        oversampling=oversampling, seed=seed, scs_hz=15e3,
        fft_size_override=fft, n_active_tones=12 * n_rb,
        cp_fraction=144 / 2048, window_fraction=1 / 64))


__all__ = ["GenOFDMConfig", "OFDMWaveform", "demodulate_ofdm",
           "ofdm_waveform", "papr_db", "wifi_waveform"]
