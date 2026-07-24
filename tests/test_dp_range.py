"""Direct-point DAC range vs the polar phase-path slew (the pi-flip
problem): range-limited DAC clips at envelope nulls, the loop's lowpass
path only recovers at loop-BW timescale, and DEVM/ACP degrade; slew-
limiting the trajectory (vector hole punching) is the fix."""
import numpy as np
import pytest

from polartx.chain import ChainConfig
from polartx.metrics.ble_metrics import bt_acp
from polartx.presets import bt_edr_adpll


@pytest.fixture(scope="module")
def wf():
    return bt_edr_adpll("8dpsk").make_waveform(n_syms=400, seed=2)


def test_raw_trajectory_needs_fs_over_2(wf):
    r = bt_edr_adpll("8dpsk").tx.run(wf, noise=False)
    req = r.info["phasemod"]["dp_required_range_hz"]
    assert req > 0.9 * wf.fs / 2          # pi flip in one sample


def test_range_limited_dac_degrades_devm_and_acp(wf):
    base = bt_edr_adpll("8dpsk", mode="event").tx.run(wf, noise=True, seed=3)
    lim = bt_edr_adpll("8dpsk", mode="event", dp_range_hz=2e6
                       ).tx.run(wf, noise=True, seed=3)
    assert lim.info["phasemod"]["dp_clip_frac"] > 0.0
    assert lim.evm()["devm_pct"] > 5.0 * base.evm()["devm_pct"]
    a = max(bt_acp(lim.y, lim.fs)["acp+2MHz_dbc"],
            bt_acp(lim.y, lim.fs)["acp-2MHz_dbc"])
    a0 = max(bt_acp(base.y, base.fs)["acp+2MHz_dbc"],
             bt_acp(base.y, base.fs)["acp-2MHz_dbc"])
    assert a > a0 + 10.0                  # ~20 dB ACP hit in practice


def test_slew_limit_makes_range_feasible(wf):
    """Trajectory-side slew limiting: required DAC range collapses to the
    limit, nothing clips, DEVM stays in class, cost is reported."""
    ch = ChainConfig(env_floor=0.05, phase_slew_max_hz=2e6)
    p = bt_edr_adpll("8dpsk", mode="event", dp_range_hz=2e6, chain=ch)
    r = p.tx.run(wf, noise=True, seed=3)
    assert r.info["phasemod"]["dp_required_range_hz"] <= 2e6 * 1.01
    assert r.info["phasemod"]["dp_clip_frac"] == 0.0
    assert r.evm()["devm_pct"] < 4.0
    assert np.isfinite(r.info["split"]["mod_evm_db"])


def test_slew_limit_complies_after_split():
    from polartx.polar import polar_split
    wfl = bt_edr_adpll("8dpsk").make_waveform(n_syms=300, seed=4)
    _, ph, info = polar_split(wfl.x, env_floor=0.05,
                              phase_slew_max_hz=2e6, fs=wfl.fs)
    slew = np.abs(np.diff(ph)) * wfl.fs / (2 * np.pi)
    assert slew.max() <= 2e6 * 1.001
    assert info["n_interp_runs"] > 0
