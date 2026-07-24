"""LTE 20 MHz narrowband polar chain: numerology, acceptance, E-UTRA
metrics."""
import numpy as np
import pytest

from polartx.metrics.aclr_ext import aclr_multi
from polartx.presets import lte20_adpll
from polartx.vendor.padpd.metrics import evm_of_signal
from polartx.waveforms.ofdm import lte_waveform


def test_lte_numerology():
    wf = lte_waveform(20e6, 64, n_symbols=2)
    cfg = wf.ofdm_ref.config
    assert cfg.fft_size == 2048 and cfg.n_active == 1200
    assert wf.fs == pytest.approx(122.88e6)
    assert evm_of_signal(wf.x, wf.ofdm_ref).db < -80.0   # loopback exact


def test_preset_meets_lte_requirements():
    """64QAM needs -22 dB EVM / 256QAM -29 dB; E-UTRA ACLR1 limit -30."""
    p = lte20_adpll(qam=256)
    wf = p.make_waveform(n_symbols=10, seed=0)
    res = p.tx.run(wf, noise=True, seed=1)
    assert res.evm().db < -40.0            # preset lands ~-53 dB
    a = aclr_multi(res.y, res.fs, 20e6)
    assert max(a["aclr1_lower_dbc"], a["aclr1_upper_dbc"]) < -45.0
    assert max(a["aclr2_lower_dbc"], a["aclr2_upper_dbc"]) < -50.0
    assert res.check_mask()[0]             # stylized E-UTRA SEM


def test_phase_slew_is_several_x_bw():
    """OFDM polar reality check: the phase path slews at several times
    the channel BW (P99 ~ 2x BW for LTE20) — why the direct DAC range is
    the hard spec of OFDM polar (contrast the EDR study, ex05)."""
    wf = lte_waveform(20e6, 64, n_symbols=6, seed=0)
    from polartx.polar import polar_split
    _, ph, _ = polar_split(wf.x, env_floor=0.05)
    fdev = np.abs(np.diff(ph)) * wf.fs / (2 * np.pi)
    assert np.percentile(fdev, 99) > 1.5 * 20e6


def test_aclr_multi_needs_enough_fs():
    wf = lte_waveform(20e6, 64, n_symbols=2, oversampling=2)
    with pytest.raises(ValueError):
        aclr_multi(wf.x, wf.fs, 20e6, offsets=(1, 2))
