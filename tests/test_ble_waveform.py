"""BLE GFSK waveform generation and deviation metrics."""
import numpy as np
import pytest

from polartx.chain import ChainConfig, PolarTX
from polartx.dpa import DPA, DPAConfig
from polartx.metrics.ble_metrics import freq_deviation
from polartx.phasemod import IdealPhaseModulator
from polartx.waveforms import gfsk_ble


def _ideal_tx():
    return PolarTX(ChainConfig(), IdealPhaseModulator(), DPA(DPAConfig(n_bits=10)))


def test_constant_envelope_and_deviation():
    wf = gfsk_ble(400, 32e6, 1e6, pattern="prbs", seed=2)
    assert np.allclose(np.abs(wf.x), 1.0)
    # peak deviation h*rate/2 = 250 kHz (long runs reach it)
    assert abs(wf.freq_ideal.max() - 250e3) / 250e3 < 0.02


def test_df1_avg_ideal_chain():
    wf = gfsk_ble(400, 32e6, 1e6, pattern="11110000")
    res = _ideal_tx().run(wf, noise=False)
    d = freq_deviation(res.y, wf)
    assert abs(d["df1_avg_hz"] - 250e3) / 250e3 < 0.05
    assert d["wrong_sign_frac"] == 0.0


def test_df2_le2m_scales():
    wf = gfsk_ble(400, 32e6, 2e6, pattern="10101010")
    res = _ideal_tx().run(wf, noise=False)
    d = freq_deviation(res.y, wf)
    # alternating pattern through the Gaussian filter: BLE spec floor is
    # df2_avg >= 0.8 * 500 kHz for LE 2M
    assert d["df2_avg_hz"] > 0.8 * 500e3
    assert d["df2_min_hz"] > 370e3


def test_min_samples_per_symbol():
    with pytest.raises(ValueError):
        gfsk_ble(100, 4e6, 1e6)
