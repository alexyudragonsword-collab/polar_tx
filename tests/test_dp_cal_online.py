"""Background two-point gain calibration (sign-sign LMS in the event
engine, Markulic-style)."""
from polartx.presets import ble_1m_adpll
from polartx.vendor.pllsim.calibration.lms import SignSignLMS


def _run(dp_cal):
    p = ble_1m_adpll(mode="event", dp_gain=1.05)
    p.tx.phasemod.dp_cal = dp_cal
    wf = p.make_waveform(n_bits=2000, seed=5)
    return p.tx.run(wf, noise=True, seed=3)


def test_converges_to_matched_gain():
    cal = SignSignLMS(init=1.05, mu=2e-5)
    r = _run(cal)
    tr = r.info["phasemod"]["sim"].cal_traces["dp_gain"]
    tail = tr[-len(tr) // 10:].mean()
    assert abs(tail - 1.0) < 0.01            # 5% error slewed out
    # EVM lands at the matched noise floor (~3%), not the 5%-error ~6.5%
    assert r.evm()["evm_pct"] < 3.8


def test_without_cal_evm_stays_degraded():
    p = ble_1m_adpll(mode="event", dp_gain=1.05)
    wf = p.make_waveform(n_bits=2000, seed=5)
    r = p.tx.run(wf, noise=True, seed=3)
    assert r.evm()["evm_pct"] > 5.0
