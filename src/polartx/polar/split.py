"""Polar decomposition: complex baseband -> (envelope, unwrapped phase).

The defining problem of wideband polar: near envelope nulls the phase
slews by ~pi in one sample, so both polar components have far wider
spectra than the composite signal ("bandwidth expansion").  Hole punching
clamps the envelope at env_floor x rms — a deliberate, bounded EVM cost
(the clamped-region error power) traded for a bounded phase-path slew.
"""
from __future__ import annotations

import numpy as np


def polar_split(x: np.ndarray, env_floor: float = 0.0
                ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Split x into (env, phase_unwrapped, info).

    env_floor > 0 clamps the envelope at env_floor * rms(x) (hole
    punching), keeping the sample's phase.  info reports the exact EVM
    cost of the clamp: error power = sum((clamp - env)^2 over clamped
    samples) / signal power.
    """
    x = np.asarray(x, dtype=complex)
    env = np.abs(x)
    rms = np.sqrt(np.mean(env ** 2))
    info = {"rms": rms, "clamped_frac": 0.0, "clamp_evm_db": -np.inf}
    if env_floor > 0.0:
        clamp = env_floor * rms
        below = env < clamp
        if below.any():
            err_pow = np.mean(np.where(below, clamp - env, 0.0) ** 2)
            info["clamped_frac"] = float(below.mean())
            info["clamp_evm_db"] = float(10 * np.log10(err_pow / rms ** 2))
        env = np.maximum(env, clamp)
    phase = np.unwrap(np.angle(x))
    return env, phase, info


def polar_recombine(env: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return np.asarray(env, dtype=float) * np.exp(1j * np.asarray(phase, dtype=float))


def occupied_bw(x: np.ndarray, fs: float, power_frac: float = 0.99) -> float:
    """Two-sided bandwidth containing power_frac of the total power."""
    n = len(x)
    spec = np.abs(np.fft.fftshift(np.fft.fft(np.asarray(x, dtype=complex)))) ** 2
    c = np.cumsum(spec) / spec.sum()
    lo = np.searchsorted(c, (1.0 - power_frac) / 2.0)
    hi = np.searchsorted(c, 1.0 - (1.0 - power_frac) / 2.0)
    return (hi - lo) * fs / n


def bandwidth_expansion(x: np.ndarray, fs: float, env_floor: float = 0.0,
                        power_frac: float = 0.99) -> dict:
    """Occupied-BW comparison composite vs envelope vs phase-modulated carrier.

    The phase path is measured on exp(j*phase) (what the phase modulator
    must actually transmit); the envelope on (env - mean) (its AC part).
    """
    env, phase, info = polar_split(x, env_floor)
    return {
        "bw_composite": occupied_bw(x, fs, power_frac),
        "bw_env": occupied_bw(env - env.mean(), fs, power_frac),
        "bw_phase": occupied_bw(np.exp(1j * phase), fs, power_frac),
        "split_info": info,
    }
