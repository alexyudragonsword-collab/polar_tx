"""Envelope (DPA amplitude) quantization: 6 dB/bit and the analytic level."""
import numpy as np

from polartx.analysis.responses import env_quant_evm_db
from polartx.chain import ChainConfig, PolarTX
from polartx.dpa import DPA, DPAConfig
from polartx.phasemod import IdealPhaseModulator
from polartx.waveforms.ofdm import wifi_waveform


def _evm_db(n_bits, wf):
    tx = PolarTX(ChainConfig(), IdealPhaseModulator(), DPA(DPAConfig(n_bits=n_bits)))
    return tx.run(wf, noise=False).evm().db


def test_six_db_per_bit():
    wf = wifi_waveform(20e6, 256, n_symbols=3, seed=4)
    e = {b: _evm_db(b, wf) for b in (6, 8, 10)}
    assert abs((e[6] - e[8]) - 12.0) < 2.0
    assert abs((e[8] - e[10]) - 12.0) < 2.0


def test_absolute_level_matches_analytic():
    wf = wifi_waveform(20e6, 256, n_symbols=3, seed=4)
    env = np.abs(wf.x)
    rms_norm = np.sqrt(np.mean(env ** 2)) / env.max()
    osr = wf.fs / wf.bw
    for b in (7, 9):
        assert abs(_evm_db(b, wf) - env_quant_evm_db(b, rms_norm, osr)) < 3.0
