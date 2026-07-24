"""Polar decomposition: complex baseband -> (envelope, unwrapped phase).

The defining problem of wideband polar: near envelope nulls the phase
slews by ~pi in one sample, so both polar components have far wider
spectra than the composite signal ("bandwidth expansion").  Hole punching
clamps the envelope at env_floor x rms — a deliberate, bounded EVM cost
(the clamped-region error power) traded for a bounded phase-path slew.
"""
from __future__ import annotations

import numpy as np

TWOPI = 2.0 * np.pi


def polar_split(x: np.ndarray, env_floor: float = 0.0,
                phase_slew_max_hz: float | None = None, fs: float = 1.0,
                phase_interp_win: int = 4
                ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Split x into (env, phase_unwrapped, info).

    env_floor > 0 clamps the envelope at env_floor * rms(x) (hole
    punching), keeping the sample's phase.

    phase_slew_max_hz (requires fs) bounds the phase-path slew — the
    direct-DAC tuning range a two-point ADPLL must cover: wherever the
    instantaneous deviation |dphi/dt|/2pi exceeds it (the pi-flips at
    envelope nulls), the unwrapped phase is linearized over the
    offending run widened by phase_interp_win samples each side, and the
    pass is iterated until the whole trajectory complies.

    info reports the EXACT EVM cost of the whole trajectory
    modification: mod_evm_db = |recombine(env', phase') - x|^2 / |x|^2.
    """
    x = np.asarray(x, dtype=complex)
    env = np.abs(x)
    rms = np.sqrt(np.mean(env ** 2))
    info = {"rms": rms, "clamped_frac": 0.0, "clamp_evm_db": -np.inf,
            "mod_evm_db": -np.inf, "n_interp_runs": 0}
    phase = np.unwrap(np.angle(x))
    if env_floor > 0.0:
        clamp = env_floor * rms
        below = env < clamp
        if below.any():
            err_pow = np.mean(np.where(below, clamp - env, 0.0) ** 2)
            info["clamped_frac"] = float(below.mean())
            info["clamp_evm_db"] = float(10 * np.log10(err_pow / rms ** 2))
        env = np.maximum(env, clamp)
    if phase_slew_max_hz is not None:
        dphi_max = TWOPI * phase_slew_max_hz / fs
        w = max(phase_interp_win, 1)
        for _ in range(8):                       # widen until compliant
            fast = np.abs(np.diff(phase)) > dphi_max
            if not fast.any():
                break
            hot = np.convolve(fast.astype(int), np.ones(2 * w + 1),
                              mode="same") > 0   # dilate +/- w
            d = np.diff(np.concatenate(([0], hot.astype(int), [0])))
            starts = np.flatnonzero(d == 1)
            ends = np.flatnonzero(d == -1)       # exclusive, on diff grid
            for s, e in zip(starts, ends):
                i0, i1 = max(s, 0), min(e, phase.size - 1)
                if i1 - i0 < 2:
                    continue
                phase[i0:i1 + 1] = np.linspace(phase[i0], phase[i1],
                                               i1 - i0 + 1)
            info["n_interp_runs"] += int(starts.size)
            w *= 2
    if info["clamped_frac"] > 0.0 or info["n_interp_runs"] > 0:
        err = env * np.exp(1j * phase) - x
        info["mod_evm_db"] = float(10 * np.log10(
            np.mean(np.abs(err) ** 2) / rms ** 2))
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
