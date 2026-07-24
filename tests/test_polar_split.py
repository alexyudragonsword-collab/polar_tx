"""Polar decomposition: identity, hole-punch cost, bandwidth expansion."""
import numpy as np

from polartx.polar import bandwidth_expansion, polar_recombine, polar_split
from polartx.waveforms.ofdm import wifi_waveform


def _wf():
    return wifi_waveform(20e6, 256, n_symbols=4, seed=1)


def test_split_recombine_identity():
    x = _wf().x
    env, phase, _ = polar_split(x)
    err = polar_recombine(env, phase) - x
    assert 10 * np.log10(np.mean(np.abs(err) ** 2) /
                         np.mean(np.abs(x) ** 2)) < -140.0


def test_hole_punch_cost_is_exact():
    x = _wf().x
    floor = 0.1
    env, phase, info = polar_split(x, env_floor=floor)
    y = polar_recombine(env, phase)
    err_db = 10 * np.log10(np.mean(np.abs(y - x) ** 2) /
                           np.mean(np.abs(x) ** 2))
    assert info["clamped_frac"] > 0.0
    assert abs(err_db - info["clamp_evm_db"]) < 0.01


def test_bandwidth_expansion_wideband_polar():
    """The 'why wideband polar is hard' numbers: both polar components
    occupy far more spectrum than the composite signal."""
    wf = _wf()
    b = bandwidth_expansion(wf.x, wf.fs, env_floor=0.02)
    assert b["bw_phase"] > 3.0 * b["bw_composite"]
    assert b["bw_env"] > 1.5 * b["bw_composite"]
