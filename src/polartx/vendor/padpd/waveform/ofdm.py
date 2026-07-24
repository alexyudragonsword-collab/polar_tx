# Vendored from PA_DPD@44f9ee99: src/padpd/waveform/ofdm.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""802.11be-style OFDM waveform generation and demodulation.

Scope: waveform-level model for PA/DPD work. All active tones carry known
random QAM data (no preamble, pilots, or channel coding). Numerology follows
802.11ax/be: 78.125 kHz subcarrier spacing, 12.8 us symbol, 0.8 us guard
interval by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .qam import qam_modulate

SUBCARRIER_SPACING_HZ = 78.125e3

# Active (data + pilot) tone counts per channel bandwidth, 802.11ax/be-like.
_ACTIVE_TONES = {
    20e6: 242,
    40e6: 484,
    80e6: 996,
    160e6: 1992,
    320e6: 3984,
}


@dataclass
class OFDMConfig:
    bandwidth_hz: float = 160e6
    qam_order: int = 1024
    n_symbols: int = 20
    oversampling: int = 4
    cp_fraction: float = 1 / 16  # 0.8 us GI over 12.8 us symbol
    window_fraction: float = 1 / 32  # raised-cosine edge, 0.4 us default
    seed: int | None = 0

    @property
    def fft_size(self) -> int:
        n = self.bandwidth_hz / SUBCARRIER_SPACING_HZ
        if abs(n - round(n)) > 1e-9:
            raise ValueError("bandwidth must be a multiple of 78.125 kHz")
        return int(round(n))

    @property
    def n_active(self) -> int:
        if self.bandwidth_hz in _ACTIVE_TONES:
            return _ACTIVE_TONES[self.bandwidth_hz]
        # Fallback for non-standard bandwidths: ~94% occupancy, even count.
        return 2 * int(0.47 * self.fft_size)

    @property
    def cp_len(self) -> int:
        return int(round(self.fft_size * self.cp_fraction))

    @property
    def window_len(self) -> int:
        w = int(round(self.fft_size * self.window_fraction))
        if w > self.cp_len:
            raise ValueError("window transition must fit inside the CP")
        return w

    @property
    def sample_rate_hz(self) -> float:
        return self.bandwidth_hz * self.oversampling

    def active_tone_indices(self) -> np.ndarray:
        """Signed tone indices, symmetric around (and excluding) DC."""
        half = self.n_active // 2
        return np.concatenate([np.arange(-half, 0), np.arange(1, half + 1)])


@dataclass
class OFDMWaveform:
    """Generated waveform plus everything needed to demodulate it."""

    x: np.ndarray  # complex baseband samples, unit average power
    tx_symbols: np.ndarray  # (n_symbols, n_active) constellation points
    config: OFDMConfig
    scale: float  # divide-by factor applied for unit-power normalization
    tone_indices: np.ndarray = field(repr=False, default=None)

    @property
    def sample_rate_hz(self) -> float:
        return self.config.sample_rate_hz


def generate_ofdm(config: OFDMConfig) -> OFDMWaveform:
    """Generate an oversampled OFDM burst with known random QAM payload."""
    rng = np.random.default_rng(config.seed)
    nfft = config.fft_size
    os_nfft = nfft * config.oversampling
    cp = config.cp_len * config.oversampling
    tones = config.active_tone_indices()

    labels = rng.integers(0, config.qam_order,
                          size=(config.n_symbols, config.n_active))
    tx_symbols = qam_modulate(labels, config.qam_order)

    # Zero-padded IFFT implements ideal oversampling.
    freq = np.zeros((config.n_symbols, os_nfft), dtype=complex)
    freq[:, tones % os_nfft] = tx_symbols
    time = np.fft.ifft(freq, axis=1) * os_nfft / np.sqrt(config.n_active)

    with_cp = np.concatenate([time[:, -cp:], time], axis=1)

    # Raised-cosine symbol windowing with overlap-add, as in real 802.11
    # transmitters, to suppress sinc sidelobes. The ramps only touch the
    # first `w` CP samples and a cyclic postfix, never the FFT window, so
    # demodulation stays exact.
    w = config.window_len * config.oversampling
    sym_len = with_cp.shape[1]
    if w > 0:
        ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(w) + 0.5) / w))
        ext = np.concatenate([with_cp, with_cp[:, cp:cp + w]], axis=1)
        ext[:, :w] *= ramp
        ext[:, sym_len:] *= ramp[::-1]
        x = np.zeros(config.n_symbols * sym_len + w, dtype=complex)
        for i in range(config.n_symbols):
            x[i * sym_len: i * sym_len + sym_len + w] += ext[i]
        x = x[: config.n_symbols * sym_len]
    else:
        x = with_cp.reshape(-1)

    scale = np.sqrt(np.mean(np.abs(x) ** 2))
    x = x / scale
    return OFDMWaveform(x=x, tx_symbols=tx_symbols, config=config,
                        scale=scale, tone_indices=tones)


def demodulate_ofdm(y: np.ndarray, ref: OFDMWaveform) -> np.ndarray:
    """Demodulate samples aligned with ``ref`` back to constellation points.

    Returns an (n_symbols, n_active) array. No equalization is applied here;
    metric functions decide how to normalize (scalar or per-tone).
    """
    cfg = ref.config
    os_nfft = cfg.fft_size * cfg.oversampling
    cp = cfg.cp_len * cfg.oversampling
    sym_len = os_nfft + cp
    n_sym = len(y) // sym_len
    if n_sym < cfg.n_symbols:
        raise ValueError("signal shorter than the reference waveform")

    blocks = y[: cfg.n_symbols * sym_len].reshape(cfg.n_symbols, sym_len)
    freq = np.fft.fft(blocks[:, cp:], axis=1)
    rx = freq[:, ref.tone_indices % os_nfft]
    # Undo generation scaling so a distortion-free loopback returns
    # tx_symbols exactly.
    rx *= ref.scale * np.sqrt(cfg.n_active) / os_nfft
    return rx


def papr_db(x: np.ndarray) -> float:
    """Peak-to-average power ratio in dB."""
    p = np.abs(x) ** 2
    return 10 * np.log10(p.max() / p.mean())
