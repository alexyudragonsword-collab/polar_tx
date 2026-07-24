"""5G NR wideband polar chains: numerology and acceptance."""
import pytest

from polartx.metrics.aclr_ext import aclr_multi
from polartx.presets import nr_dtc
from polartx.vendor.padpd.metrics import evm_of_signal
from polartx.waveforms.ofdm import nr_waveform


def test_nr_numerologies_loopback():
    for bw, scs, fft, tones in ((100e6, 30e3, 4096, 3276),
                                (200e6, 120e3, 2048, 1584)):
        wf = nr_waveform(bw, scs, 64, n_symbols=2)
        cfg = wf.ofdm_ref.config
        assert cfg.fft_size == fft and cfg.n_active == tones
        assert evm_of_signal(wf.x, wf.ofdm_ref).db < -80.0


def test_fr1_100mhz_meets_256qam():
    """NR 256-QAM needs -29 dB EVM; FR1 preset lands ~-37 dB."""
    p = nr_dtc(bw=100e6)
    res = p.tx.run(p.make_waveform(n_symbols=4, seed=0), noise=True, seed=1)
    assert res.evm().db < -33.0
    a = aclr_multi(res.y, res.fs, 100e6, offsets=(1,))
    assert max(a["aclr1_lower_dbc"], a["aclr1_upper_dbc"]) < -45.0
    assert res.check_mask()[0]


def test_fr2_200mhz_meets_64qam():
    """FR2 @ 28 GHz is mmWave-LO-limited; 64-QAM needs -22 dB."""
    p = nr_dtc(bw=200e6)
    res = p.tx.run(p.make_waveform(n_symbols=4, seed=0), noise=True, seed=1)
    assert res.evm().db < -24.0
    assert res.check_mask()[0]


def test_bad_numerology_rejected():
    with pytest.raises(ValueError):
        nr_waveform(200e6, 30e3, 64)
