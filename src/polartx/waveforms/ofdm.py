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

    @property
    def fft_size(self) -> int:  # type: ignore[override]
        n = self.bandwidth_hz / self.scs_hz
        if abs(n - round(n)) > 1e-9:
            raise ValueError("bandwidth must be a multiple of scs_hz")
        return int(round(n))

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


__all__ = ["GenOFDMConfig", "OFDMWaveform", "demodulate_ofdm",
           "ofdm_waveform", "papr_db", "wifi_waveform"]
