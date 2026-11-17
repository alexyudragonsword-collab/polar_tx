"""Cartesian ILA-GMP memory DPD around the whole polar chain."""
import numpy as np

from polartx.cal.memory_dpd import _chain_as_pa, fit_chain_ila, run_with_ila
from polartx.chain import ChainConfig, PolarTX
from polartx.dpa import DPA, DPAConfig
from polartx.phasemod import IdealPhaseModulator
from polartx.vendor.padpd.dpd import ILAPredistorter
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
    what ILA can do — the reason ChainConfig.fs_scale_fixed exists.

    Compared against the same chain with the full scale pinned, not
    against a number: the cap moved when the vendored ILA started keeping
    its best iteration (PA_DPD d5750c2 / 70d6cc3) -- the fixed two-pass
    loop kept a diverged second pass and read ~4 dB; keep-best stops at
    pass 1 and reads 14.7 dB, still 34 dB short of the static chain."""
    wf = wifi_waveform(80e6, 256, n_symbols=4, seed=1)

    def ila_gain(tx):
        r0 = tx.run(wf, noise=False)
        dpd = fit_chain_ila(tx, wf)
        return r0.evm().db - run_with_ila(tx, wf, dpd, noise=False).evm().db

    gain_nonstatic = ila_gain(_chain(wf, None))     # per-run normalization
    gain_static = ila_gain(_chain(wf, 1.3 * np.abs(wf.x).max()))
    assert gain_nonstatic < gain_static - 20.0       # measured 14.7 vs 48.5 dB


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
    assert r1.evm().db < r0.evm().db - 15.0          # measured -19.75 -> -40.4
    a0 = aclr(r0.y, r0.fs, wf.bw)["upper_dbc"]
    a1 = aclr(r1.y, r1.fs, wf.bw)["upper_dbc"]
    assert a1 < a0                                   # measured -38.0 -> -39.1
    # Correction 2026-11-17: this used to assert the opposite (-38.0 -> -33.8, "the
    # ILA trades ACLR for EVM").  That was the second ILA pass diverging: its check
    # NMSE is 4 dB worse than the first's, and the fixed two-pass loop kept it.  The
    # vendored ILA now keeps the best pass within 1 dB (PA_DPD d5750c2 / 70d6cc3):
    assert dpd.best_iteration == 1
    assert dpd.history[2] > dpd.history[1] + dpd.keep_best_tol_db   # -34.0 -> -30.0
    # and the old fixed-count loop still gives the old trade, so the change is the
    # pass that is kept, not the model or the chain
    old = ILAPredistorter(model_factory=gmp_opendpd_510, n_iterations=2,
                          fit_kwargs={"regularization": 1e-9}, keep_best=False)
    old.fit(_chain_as_pa(tx, wf, noise=False, seed=0), wf.x)
    r_old = run_with_ila(tx, wf, old, noise=False)
    assert aclr(r_old.y, r_old.fs, wf.bw)["upper_dbc"] > a0                # -33.8
    assert r_old.evm().db > r1.evm().db + 3.0                              # -36.5 vs -40.4
