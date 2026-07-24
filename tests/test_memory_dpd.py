"""Cartesian ILA-GMP memory DPD around the whole polar chain."""
import numpy as np

from polartx.cal.memory_dpd import fit_chain_ila, run_with_ila
from polartx.chain import ChainConfig, PolarTX
from polartx.dpa import DPA, DPAConfig
from polartx.phasemod import IdealPhaseModulator
from polartx.vendor.padpd.metrics import aclr
from polartx.waveforms.ofdm import wifi_waveform


def _chain(wf, fixed_scale):
    fir = np.array([1.0, 0.12 + 0.04j])

    def memory(y):
        lin = np.convolve(y, fir, mode="full")[:y.size]
        return lin + 0.05 * lin * np.abs(lin) ** 2

    return PolarTX(
        ChainConfig(env_floor=0.02, fs_scale_fixed=fixed_scale),
        IdealPhaseModulator(),
        DPA(DPAConfig(n_bits=11, amam=("rapp", 2.5, 1.2),
                      ampm_deg_poly=(0.0, 3.0, 4.0))),
        memory=memory)


def test_ila_linearizes_chain_with_memory():
    wf = wifi_waveform(80e6, 256, n_symbols=4, seed=1)
    tx = _chain(wf, 1.3 * np.abs(wf.x).max())
    r0 = tx.run(wf, noise=False)
    dpd = fit_chain_ila(tx, wf)
    r1 = run_with_ila(tx, wf, dpd, noise=False)
    assert r1.evm().db < r0.evm().db - 20.0          # measured: -20 -> -69
    a0 = aclr(r0.y, r0.fs, wf.bw)["upper_dbc"]
    a1 = aclr(r1.y, r1.fs, wf.bw)["upper_dbc"]
    assert a1 < a0 - 15.0                            # measured: -27 -> -54


def test_fixed_full_scale_is_required():
    """Per-run peak normalization makes the chain non-static and caps
    what ILA can do — the reason ChainConfig.fs_scale_fixed exists."""
    wf = wifi_waveform(80e6, 256, n_symbols=4, seed=1)
    tx = _chain(wf, None)                            # per-run normalization
    r0 = tx.run(wf, noise=False)
    dpd = fit_chain_ila(tx, wf)
    r1 = run_with_ila(tx, wf, dpd, noise=False)
    gain_nonstatic = r0.evm().db - r1.evm().db
    assert gain_nonstatic < 15.0                     # measured: ~4 dB only
