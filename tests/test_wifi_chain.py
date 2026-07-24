"""Wideband WiFi polar chains: preset-level acceptance numbers."""
import pytest

from polartx.dpa import DPAConfig
from polartx.presets import wifi_dtc


@pytest.mark.parametrize("bw,qam,evm_req_db", [
    (80e6, 1024, -35.0),      # 802.11ax 1024-QAM needs -35 dB
    (160e6, 1024, -35.0),
    (320e6, 4096, -38.0),     # 802.11be 4096-QAM needs -38 dB
])
def test_preset_meets_evm_and_mask(bw, qam, evm_req_db):
    p = wifi_dtc(bw=bw, qam=qam)
    wf = p.make_waveform(n_symbols=4, seed=0)
    res = p.tx.run(wf, noise=True, seed=1)
    assert res.evm().db < evm_req_db
    a = res.aclr()
    assert max(a["lower_dbc"], a["upper_dbc"]) < -45.0
    assert res.check_mask()[0]


def test_ideal_floor():
    """12-bit DPA + no CFR + noiseless: the chain floor is far below any
    impairment under study."""
    p = wifi_dtc(bw=160e6, dpa=DPAConfig(n_bits=12), cfr_papr_db=None,
                 n_bits=14, dither=False, env_floor=0.0)
    wf = p.make_waveform(n_symbols=4, seed=0)
    assert p.tx.run(wf, noise=False).evm().db < -60.0
