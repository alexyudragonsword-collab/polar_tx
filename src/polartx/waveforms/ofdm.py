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
    dft_precode: bool = False               # SC-FDMA / DFT-s-OFDM uplink
    n_pilots: int = 0                       # known BPSK pilot tones
    preamble_symbols: int = 0               # known full-occupancy training
                                            # symbols prepended (channel est)

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


def _synth_grid(cfg, sym: np.ndarray):
    """Frequency-domain symbol grid -> time-domain burst.

    Adapted from the vendored padpd generate_ofdm synthesis (IFFT
    oversampling, CP, raised-cosine overlap-add) so precoded/pilot/
    preamble grids share the exact same math; the no-feature path is
    regression-tested bit-exact against the vendored generator."""
    from dataclasses import replace

    n_sym = sym.shape[0]
    cfg2 = replace(cfg, n_symbols=n_sym)
    nfft = cfg2.fft_size
    os_nfft = nfft * cfg2.oversampling
    cp = cfg2.cp_len * cfg2.oversampling
    tones = cfg2.active_tone_indices()

    freq = np.zeros((n_sym, os_nfft), dtype=complex)
    freq[:, tones % os_nfft] = sym
    time = np.fft.ifft(freq, axis=1) * os_nfft / np.sqrt(cfg2.n_active)
    with_cp = np.concatenate([time[:, -cp:], time], axis=1)
    w = cfg2.window_len * cfg2.oversampling
    sym_len = with_cp.shape[1]
    if w > 0:
        ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(w) + 0.5) / w))
        ext = np.concatenate([with_cp, with_cp[:, cp:cp + w]], axis=1)
        ext[:, :w] *= ramp
        ext[:, sym_len:] *= ramp[::-1]
        x = np.zeros(n_sym * sym_len + w, dtype=complex)
        for i in range(n_sym):
            x[i * sym_len: i * sym_len + sym_len + w] += ext[i]
        x = x[: n_sym * sym_len]
    else:
        x = with_cp.reshape(-1)
    scale = np.sqrt(np.mean(np.abs(x) ** 2))
    return OFDMWaveform(x=x / scale, tx_symbols=sym, config=cfg2,
                        scale=scale, tone_indices=tones)


def ofdm_waveform(cfg: GenOFDMConfig) -> Waveform:
    """Generate and wrap into the chain's Waveform container.

    Beyond the vendored engine: dft_precode (SC-FDMA/DFT-s-OFDM
    uplink), n_pilots (known BPSK pilot tones for CPE tracking) and
    preamble_symbols (known full-occupancy training symbols for
    channel estimation).  Channel coding is deliberately out of scope:
    no TX-impairment metric is measured downstream of the bit mapping.
    """
    meta = {"qam": cfg.qam_order, "scs_hz": cfg.scs_hz,
            "dft_precode": cfg.dft_precode, "n_pilots": cfg.n_pilots,
            "preamble_symbols": cfg.preamble_symbols}
    plain = (not cfg.dft_precode and cfg.n_pilots == 0
             and cfg.preamble_symbols == 0)
    if plain:
        ref = generate_ofdm(cfg)
        meta["papr_db"] = papr_db(ref.x)
        return Waveform(x=ref.x, fs=cfg.sample_rate_hz,
                        bw=cfg.bandwidth_hz, kind="ofdm", ofdm_ref=ref,
                        meta=meta)

    if cfg.dft_precode and cfg.n_pilots:
        raise ValueError("SC-FDMA uses full pilot symbols (preamble_"
                         "symbols), not scattered pilot tones")
    from ..vendor.padpd.waveform.qam import qam_modulate as _qam
    rng = np.random.default_rng(cfg.seed)
    n_act = cfg.n_active

    # data grid
    labels = rng.integers(0, cfg.qam_order, size=(cfg.n_symbols, n_act))
    data = _qam(labels, cfg.qam_order)
    meta["qam_symbols"] = data.copy()          # pre-precoding, for plots
    pilot_idx = np.array([], dtype=int)
    if cfg.n_pilots:
        pilot_idx = np.linspace(0, n_act - 1, cfg.n_pilots + 2,
                                dtype=int)[1:-1]
        prng = np.random.default_rng(cfg.seed + 777)
        pilots = 1.0 - 2.0 * prng.integers(0, 2, (cfg.n_symbols,
                                                  cfg.n_pilots))
        data[:, pilot_idx] = pilots
    if cfg.dft_precode:
        data = np.fft.fft(data, axis=1) / np.sqrt(n_act)

    if cfg.preamble_symbols:
        prng = np.random.default_rng(cfg.seed + 999)
        pre = 1.0 - 2.0 * prng.integers(0, 2, (cfg.preamble_symbols,
                                               n_act)).astype(float)
        grid = np.concatenate([pre.astype(complex), data], axis=0)
    else:
        grid = data
    ref = _synth_grid(cfg, grid)
    meta.update({"papr_db": papr_db(ref.x), "pilot_idx": pilot_idx})
    return Waveform(x=ref.x, fs=cfg.sample_rate_hz, bw=cfg.bandwidth_hz,
                    kind="ofdm", ofdm_ref=ref, meta=meta)


#: stylized 802.11ax/be pilot counts per bandwidth
WIFI_PILOTS = {20e6: 8, 40e6: 16, 80e6: 16, 160e6: 32, 320e6: 64}


def wifi_waveform(bw: float = 160e6, qam: int = 1024, *, n_symbols: int = 8,
                  oversampling: int = 4, seed: int = 0,
                  pilots: bool = False, preamble: int = 0) -> Waveform:
    """802.11ax/be-style channel.

    Default (pilots=False, preamble=0) is bit-exact vs the vendored
    generator.  pilots=True embeds the stylized per-BW pilot count for
    receiver-style CPE tracking; preamble prepends known full-occupancy
    training symbols for channel estimation (metrics.ofdm_rx.evm_rx)."""
    return ofdm_waveform(GenOFDMConfig(
        bandwidth_hz=bw, qam_order=qam, n_symbols=n_symbols,
        oversampling=oversampling, seed=seed,
        n_pilots=WIFI_PILOTS.get(bw, 8) if pilots else 0,
        preamble_symbols=preamble))


#: LTE channel BW -> (FFT size @ 15 kHz SCS, resource blocks)
LTE_NUMEROLOGY = {1.4e6: (128, 6), 3e6: (256, 15), 5e6: (512, 25),
                  10e6: (1024, 50), 15e6: (1536, 75), 20e6: (2048, 100)}

#: 5G NR (channel BW, SCS) -> (FFT size, resource blocks), 38.104-style
NR_NUMEROLOGY = {(50e6, 30e3): (2048, 133), (100e6, 30e3): (4096, 273),
                 (100e6, 120e3): (1024, 66), (200e6, 120e3): (2048, 132)}


def lte_waveform(bw: float = 20e6, qam: int = 64, *, n_symbols: int = 28,
                 oversampling: int = 4, seed: int = 0,
                 sc_fdma: bool = False) -> Waveform:
    """E-UTRA-style channel: 15 kHz SCS, standard FFT/RB table (20 MHz:
    2048-FFT / 1200 tones, fs = 30.72 * os MS/s), normal-CP-like
    144/2048 guard.  sc_fdma=True DFT-precodes each symbol — the real
    UPLINK waveform (~1.7 dB lower PAPR, what a handset polar TX
    actually transmits).  Stylized: data tones only, no DMRS/PSS."""
    if bw not in LTE_NUMEROLOGY:
        raise ValueError(f"bw must be one of {sorted(LTE_NUMEROLOGY)}")
    fft, n_rb = LTE_NUMEROLOGY[bw]
    return ofdm_waveform(GenOFDMConfig(
        bandwidth_hz=bw, qam_order=qam, n_symbols=n_symbols,
        oversampling=oversampling, seed=seed, scs_hz=15e3,
        fft_size_override=fft, n_active_tones=12 * n_rb,
        cp_fraction=144 / 2048, window_fraction=1 / 64,
        dft_precode=sc_fdma))


def nr_waveform(bw: float = 100e6, scs: float = 30e3, qam: int = 256, *,
                n_symbols: int = 8, oversampling: int = 4,
                seed: int = 0, dft_precode: bool = False) -> Waveform:
    """5G-NR-style CP-OFDM channel: 38.104 FFT/RB table (100 MHz @ 30 kHz:
    4096-FFT / 3276 tones; 200 MHz @ 120 kHz FR2: 2048-FFT / 1584 tones).
    Stylized: data tones only, no DMRS/SSB."""
    key = (bw, scs)
    if key not in NR_NUMEROLOGY:
        raise ValueError(f"(bw, scs) must be one of {sorted(NR_NUMEROLOGY)}")
    fft, n_rb = NR_NUMEROLOGY[key]
    return ofdm_waveform(GenOFDMConfig(
        bandwidth_hz=bw, qam_order=qam, n_symbols=n_symbols,
        oversampling=oversampling, seed=seed, scs_hz=scs,
        fft_size_override=fft, n_active_tones=12 * n_rb,
        cp_fraction=144 / 2048, window_fraction=1 / 64,
        dft_precode=dft_precode))


__all__ = ["GenOFDMConfig", "OFDMWaveform", "demodulate_ofdm",
           "lte_waveform", "nr_waveform", "ofdm_waveform", "papr_db",
           "wifi_waveform"]
