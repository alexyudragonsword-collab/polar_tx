"""Polar DPD (AM-AM/AM-PM inverse LUTs): exact inversion and measured fit."""
import numpy as np

from polartx.cal.polar_dpd import PolarDPD
from polartx.chain import ChainConfig, PolarTX
from polartx.dpa import DPA, DPAConfig
from polartx.phasemod import IdealPhaseModulator
from polartx.waveforms.ofdm import lte_waveform

DPA_CFG = DPAConfig(n_bits=10, n_thermo=6, amam=("rapp", 2.5, 1.1),
                    ampm_deg_poly=(0.0, 2.0, 3.0))


def _run(dpd):
    wf = lte_waveform(20e6, 64, n_symbols=6, seed=1)
    tx = PolarTX(ChainConfig(env_floor=0.05), IdealPhaseModulator(),
                 DPA(DPA_CFG), dpd=dpd)
    return tx.run(wf, noise=False)


def test_from_dpa_linearizes():
    e0 = _run(None).evm().db
    e1 = _run(PolarDPD.from_dpa(DPA(DPA_CFG))).evm().db
    assert e1 < e0 - 12.0          # >= 12 dB EVM improvement
    assert e1 < -50.0


def test_fit_from_measurement_close_to_exact():
    """The binned-observation fit lands within a few dB of the exact
    model inversion."""
    res0 = _run(None)
    dpd_fit = PolarDPD.fit(res0)
    e_fit = _run(dpd_fit).evm().db
    e_exact = _run(PolarDPD.from_dpa(DPA(DPA_CFG))).evm().db
    assert e_fit < res0.evm().db - 10.0
    assert abs(e_fit - e_exact) < 6.0


def test_luts_monotone():
    dpd = PolarDPD.from_dpa(DPA(DPA_CFG))
    assert (np.diff(dpd.amp_out) >= -1e-12).all()
