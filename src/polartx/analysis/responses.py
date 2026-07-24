"""Analytic companions to the time-domain chain: closed-form levels the
tests and design charts check the simulation against."""
from __future__ import annotations

import numpy as np

TWOPI = 2.0 * np.pi


def dtc_quant_phase_rms(n_bits: int, range_ui: float = 1.0,
                        osr: float = 1.0, dither_order: int = 0) -> float:
    """In-band rms phase error [rad] of the DTC phase quantizer.

    Uniform quantization noise q^2/12 spread white over fs; the in-band
    fraction is 1/osr (osr = fs/bw).  First-order error-feedback dither
    shapes it by |2 sin(pi f / fs)|^2: in-band power multiplies by
    (2/3) * pi^2 / osr^2 for osr >> 1.
    """
    q = TWOPI * range_ui / (1 << n_bits)
    var = q * q / 12.0 / osr
    if dither_order == 1:
        var *= (np.pi / osr) ** 2 * 2.0 / 3.0 * osr  # integral of 4sin^2 over band
    return float(np.sqrt(var))


def evm_db_from_phase_rms(phase_rms_rad: float) -> float:
    """Small-angle: constellation EVM equals the in-band phase error rms."""
    return float(20.0 * np.log10(max(phase_rms_rad, 1e-15)))


def inl_sin_spur_dbc(amp_ui: float) -> float:
    """Spur level for a sinusoidal INL term under a CW frequency-offset
    stimulus: phase error amp * sin(2 pi k * code(t)) is a phase tone of
    amplitude 2*pi*amp_ui rad -> sidebands at 20 log10(amp_rad / 2)."""
    return float(20.0 * np.log10(TWOPI * amp_ui / 2.0))


def zoh_image_dbc(f_sig: float, f_update: float) -> float:
    """Level of the first ZOH replica (at f_update - f_sig) relative to
    the held tone at f_sig, both shaped by the ZOH sinc."""
    def _s(f):
        x = f / f_update
        return np.sinc(x)
    return float(20.0 * np.log10(abs(_s(f_update - f_sig)) /
                                 max(abs(_s(f_sig)), 1e-15)))


def env_quant_evm_db(n_bits: int, env_rms_norm: float,
                     osr: float = 1.0) -> float:
    """In-band EVM of B-bit envelope quantization: LSB^2/12 error power
    spread over fs, in-band fraction 1/osr, referenced to signal power
    (env_rms_norm = rms envelope / full scale)."""
    lsb = 1.0 / ((1 << n_bits) - 1)
    var = lsb * lsb / 12.0 / osr
    return float(10.0 * np.log10(var / env_rms_norm ** 2))
