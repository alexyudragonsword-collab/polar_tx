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


def test_ila_linearizes_the_measured_dpa_with_its_residual_memory():
    """The real device's memory, inverted from the OpenDPD capture, in the
    chain: the whole-chain ILA (given the residual's own GMP-510
    structure) must still linearize it by >= 15 dB.  needs the data."""
    import pytest
    from dataclasses import replace
    from polartx.measured import find_opendpd_root, load_measured_dpa
    from polartx.vendor.padpd.pa import gmp_opendpd_510
    if find_opendpd_root() is None:
        pytest.skip("OpenDPD dataset clone not found")
    dpa, ch = load_measured_dpa("DPA_160MHz", with_memory=True)
    wf = wifi_waveform(160e6, 1024, n_symbols=4, seed=1)
    wf = replace(wf, x=wf.x / np.abs(wf.x).max() * ch["fs_scale"])   # the capture's drive
    assert abs(wf.fs - ch["fs"]) < 1.0                                 # and its sample rate
    tx = PolarTX(ChainConfig(env_floor=0.02, fs_scale_fixed=ch["fs_scale"]),
                 IdealPhaseModulator(), dpa, memory=ch["memory"])
    r0 = tx.run(wf, noise=False)
    dpd = fit_chain_ila(tx, wf, model_factory=gmp_opendpd_510)
    r1 = run_with_ila(tx, wf, dpd, noise=False)
    assert r1.evm().db < r0.evm().db - 15.0          # measured -19.75 -> -36.5
    a0 = aclr(r0.y, r0.fs, wf.bw)["upper_dbc"]
    a1 = aclr(r1.y, r1.fs, wf.bw)["upper_dbc"]
    assert a1 > a0                                   # measured -38.0 -> -33.8: the ILA trades
    # ACLR for EVM here (ex09 shows the same); the memory's spectral regrowth is NOT
    # reduced by this inverse -- recorded, not hidden
