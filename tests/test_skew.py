"""AM/PM path skew: EVM impact, estimation accuracy, correction."""
import numpy as np
import pytest

from polartx.cal import corrected_chain_config, estimate_env_skew
from polartx.presets import wifi_dtc

BW = 160e6


@pytest.fixture(scope="module")
def wf():
    return wifi_dtc(bw=BW).make_waveform(n_symbols=4, seed=2)


def test_skew_degrades_evm_monotonically(wf):
    e = [wifi_dtc(bw=BW, env_skew_s=t).tx.run(wf, noise=False).evm().db
         for t in (0.0, 1e-9, 2e-9, 4e-9)]
    assert all(b > a + 2.0 for a, b in zip(e, e[1:]))


def test_skew_estimation_accuracy(wf):
    fs = BW * 4
    for skew in (0.8e-9, 2.3e-9, -1.5e-9):
        res = wifi_dtc(bw=BW, env_skew_s=skew).tx.run(wf, noise=False)
        est = estimate_env_skew(res)
        assert abs(est["skew_samples"] - skew * fs) < 0.05


def test_skew_correction_restores_evm(wf):
    base = wifi_dtc(bw=BW).tx.run(wf, noise=False).evm().db
    p = wifi_dtc(bw=BW, env_skew_s=2e-9)
    res = p.tx.run(wf, noise=False)
    est = estimate_env_skew(res)
    p.tx.cfg = corrected_chain_config(p.tx.cfg, est["skew_s"])
    fixed = p.tx.run(wf, noise=False).evm().db
    assert fixed < base + 1.0
