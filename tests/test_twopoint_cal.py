"""Offline two-point gain estimator: one-shot LS recovery of eps."""
import pytest

from polartx.cal.twopoint import estimate_dp_gain_error
from polartx.presets import ble_1m_adpll, lte20_adpll


@pytest.mark.parametrize("eps", [0.01, 0.03, -0.02])
def test_estimator_recovers_eps_noiseless(eps):
    p = lte20_adpll(qam=64, dp_gain=1.0 + eps)
    wf = p.make_waveform(n_symbols=6, seed=0)
    r = p.tx.run(wf, noise=False)
    est = estimate_dp_gain_error(p.tx.phasemod, r.phase_cmd, r.phase_out,
                                 r.fs)
    assert est["eps_hat"] == pytest.approx(eps, rel=0.02)


def test_correction_restores_evm_with_noise():
    p = lte20_adpll(qam=64, dp_gain=1.03)
    wf = p.make_waveform(n_symbols=10, seed=0)
    r = p.tx.run(wf, noise=True, seed=2)
    est = estimate_dp_gain_error(p.tx.phasemod, r.phase_cmd, r.phase_out,
                                 r.fs)
    p.tx.phasemod.dp_gain *= est["dp_gain_corr"]
    r2 = p.tx.run(wf, noise=True, seed=2)
    assert r2.evm().db < -50.0 and r2.evm().db < r.evm().db - 25.0


def test_works_on_gfsk_too():
    p = ble_1m_adpll(dp_gain=1.05)
    wf = p.make_waveform(n_bits=600, seed=5)
    r = p.tx.run(wf, noise=False)
    est = estimate_dp_gain_error(p.tx.phasemod, r.phase_cmd, r.phase_out,
                                 r.fs)
    assert est["eps_hat"] == pytest.approx(0.05, rel=0.05)
