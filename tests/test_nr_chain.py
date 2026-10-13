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


def test_nr_400mhz_fr2_numerology_and_preset_rate():
    """TS 38.101-2 Table 5.3.2-1: 400 MHz @ 120 kHz is 264 RB (3168 tones on
    a 4096 FFT); the preset's fs_bb is the waveform's own rate, fft * scs *
    oversampling, not bw * oversampling."""
    import numpy as np
    from polartx import nr_dtc
    from polartx.waveforms.ofdm import NR_NUMEROLOGY, nr_waveform
    assert NR_NUMEROLOGY[(400e6, 120e3)] == (4096, 264)
    assert NR_NUMEROLOGY[(50e6, 120e3)] == (512, 32)
    wf = nr_waveform(400e6, 120e3, 64, n_symbols=1)
    assert wf.fs == 4096 * 120e3 * 4 and wf.ofdm_ref.config.n_active == 3168
    assert abs(np.mean(np.abs(wf.x) ** 2) - 1.0) < 1e-9
    p = nr_dtc(bw=400e6)
    assert p.fs_bb == wf.fs == p.make_waveform(n_symbols=1).fs
    p1 = nr_dtc()                                   # FR1 100 MHz @ 30 kHz
    assert p1.fs_bb == 4096 * 30e3 * 4 == p1.make_waveform(n_symbols=1).fs   # 491.52 MS/s, not 400
    import pytest
    with pytest.raises(ValueError, match="nr_dtc"):
        nr_dtc(bw=300e6)
