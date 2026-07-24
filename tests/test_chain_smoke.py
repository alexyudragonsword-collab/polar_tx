"""Chain plumbing: ideal blocks give an essentially perfect BLE TX."""
import numpy as np

from polartx.chain import ChainConfig, PolarTX
from polartx.dpa import DPA, DPAConfig
from polartx.phasemod import IdealPhaseModulator
from polartx.waveforms import gfsk_ble


def test_ideal_chain_is_transparent():
    wf = gfsk_ble(300, 32e6, 1e6, seed=4)
    tx = PolarTX(ChainConfig(), IdealPhaseModulator(), DPA(DPAConfig(n_bits=10)))
    res = tx.run(wf, noise=False)
    assert res.evm()["evm_db"] < -60.0
    # constant envelope -> single DPA code
    assert np.unique(res.env_code).size == 1
    ok, margin, _ = res.check_mask()
    assert ok


def test_result_taps_shapes():
    wf = gfsk_ble(200, 32e6, 1e6)
    tx = PolarTX(ChainConfig(), IdealPhaseModulator(), DPA(DPAConfig()))
    res = tx.run(wf, noise=False)
    n = wf.n
    assert res.y.size == n and res.phase_out.size == n
    assert res.env_cmd.size == n and res.env_code.size == n
