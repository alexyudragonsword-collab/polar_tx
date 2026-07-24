"""DPA supply pushing: envelope-correlated PM that polar LUTs can't fix."""
import numpy as np

from polartx.chain import ChainConfig, PolarTX, SupplyConfig
from polartx.dpa import DPA, DPAConfig
from polartx.phasemod import IdealPhaseModulator
from polartx.waveforms.ofdm import wifi_waveform


def _run(k_push, dpd=None):
    wf = wifi_waveform(80e6, 256, n_symbols=4, seed=1)
    sup = SupplyConfig(k_push_hz_v=k_push) if k_push else None
    tx = PolarTX(ChainConfig(env_floor=0.02, supply=sup),
                 IdealPhaseModulator(), DPA(DPAConfig(n_bits=11)), dpd=dpd)
    return tx.run(wf, noise=False)


def test_pushing_degrades_evm_linearly():
    e1 = _run(2e6).evm().db
    e2 = _run(4e6).evm().db
    e0 = _run(0).evm().db
    assert e1 > e0 + 10.0                     # pushing is a real impairment
    assert abs((e2 - e1) - 6.0) < 1.5         # phase error linear in k_push


def test_static_polar_dpd_cannot_fix_it():
    """The ripple filter gives the AM->PM memory: a code-static LUT DPD
    leaves the EVM essentially unchanged."""
    from polartx.cal.polar_dpd import PolarDPD
    dpa = DPA(DPAConfig(n_bits=11))
    e_raw = _run(4e6).evm().db
    e_dpd = _run(4e6, dpd=PolarDPD.from_dpa(dpa)).evm().db
    assert abs(e_dpd - e_raw) < 2.0
