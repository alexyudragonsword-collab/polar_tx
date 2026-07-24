"""DPA interleaving (amplitude-image comb) and the post-DPA memory hook."""
import numpy as np

from polartx.chain import ChainConfig, PolarTX
from polartx.dpa import DPA, DPAConfig
from polartx.impairments import zoh_hold
from polartx.phasemod import IdealPhaseModulator
from polartx.waveforms.ofdm import wifi_waveform


def _am_tone_image_dbc(interleave, hold=8, f_am_bin=64):
    """Block-level: AM tone on a carrier, DPA codes held at fs/hold with
    N staggered banks; return the level of the first amplitude image at
    f_dpa - f_am."""
    n = 1 << 14
    dpa = DPA(DPAConfig(n_bits=12))
    env = 0.7 + 0.2 * np.sin(2 * np.pi * f_am_bin * np.arange(n) / n)
    code = dpa.encode(env)
    banks = [zoh_hold(code, hold, k * (hold // interleave))
             for k in range(interleave)]
    y = np.mean([dpa(b, np.zeros(n)) for b in banks], axis=0)
    s = np.abs(np.fft.fft(y * np.hanning(n)))
    db = 20 * np.log10(np.maximum(s / s.max(), 1e-12))
    img = n // hold - f_am_bin                 # f_dpa - f_am
    return db[img - 2: img + 3].max()


def test_interleave_combs_out_first_image():
    i1 = _am_tone_image_dbc(1)
    i2 = _am_tone_image_dbc(2)
    i4 = _am_tone_image_dbc(4)
    assert i2 < i1 - 15.0 and i4 < i1 - 15.0


def test_interleave_improves_chain_evm():
    wf = wifi_waveform(80e6, 256, n_symbols=3, seed=1)
    e = []
    for il in (1, 2, 4):
        tx = PolarTX(ChainConfig(env_floor=0.02, f_dpa=wf.fs / 8,
                                 interleave=il),
                     IdealPhaseModulator(), DPA(DPAConfig(n_bits=12)))
        e.append(tx.run(wf, noise=False).evm().db)
    assert e[1] < e[0] - 1.0 and e[2] < e[1]


def test_memory_hook_linear_fir():
    """Linear memory shows in scalar-equalized EVM but is absorbed by a
    per-tone equalizer — the standard linear-vs-nonlinear separation."""
    wf = wifi_waveform(80e6, 256, n_symbols=3, seed=1)
    fir = np.array([1.0, 0.15 + 0.05j])
    tx = PolarTX(ChainConfig(env_floor=0.02), IdealPhaseModulator(),
                 DPA(DPAConfig(n_bits=12)),
                 memory=lambda y: np.convolve(y, fir, mode="full")[:y.size])
    res = tx.run(wf, noise=False)
    assert res.evm(equalize="scalar").db > -30.0
    assert res.evm(equalize="per_tone").db < -60.0
