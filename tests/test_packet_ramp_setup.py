"""Tier 3: EDR full packet, power ramping, setup serialization."""
import numpy as np
import pytest

from polartx.impairments import apply_ramp
from polartx.metrics.dpsk import packet_metrics
from polartx.presets import bt_edr_adpll
from polartx.waveforms.edr import edr_packet


def test_edr_packet_through_chain():
    """GFSK header + guard + DPSK payload in one burst: header delta-f
    and payload DEVM both healthy, envelope actually switches regimes."""
    p = bt_edr_adpll("8dpsk")
    wf = edr_packet(300, 32e6, mode="8dpsk", seed=2)
    res = p.tx.run(wf, noise=True, seed=3)
    m = packet_metrics(res.y, wf)
    assert abs(m["header_dev_avg_hz"] - 250e3) / 250e3 < 0.2
    assert m["header_wrong_sign_frac"] == 0.0
    assert m["payload_devm_pct"] < 6.5              # limit 13%
    g0, g1 = wf.meta["segments"]["gfsk"]
    d0, d1 = wf.meta["segments"]["dpsk"]
    assert np.std(np.abs(wf.x[g0:g1])) < 1e-6       # const-env header
    assert np.std(np.abs(wf.x[d0:d1])) > 0.1        # modulated payload


def test_fast_ramp_splatters_transient_acp():
    """Ramp specs are MAX-HOLD power-vs-time in the adjacent channel —
    a Welch average hides the keying transient entirely (its Hann taper
    kills the burst edges; measured: identical steady ACP for any
    ramp).  Max-hold shows the physics: a hard-keyed burst transients
    at -20 dBc @ 2 MHz vs -56 dBc with a 2 us raised-cosine ramp."""
    from polartx.metrics.ble_metrics import acp_transient_db
    p = bt_edr_adpll("8dpsk")
    wf = p.make_waveform(n_syms=200, seed=2)
    y0 = p.tx.run(wf, noise=False).y
    hard = acp_transient_db(apply_ramp(y0, wf.fs, 0.0), wf.fs)
    soft = acp_transient_db(apply_ramp(y0, wf.fs, 2e-6), wf.fs)
    assert hard > soft + 25.0
    assert soft < -50.0


def test_setup_roundtrip(tmp_path):
    from polartx.guiutil import load_setup, run_setup, save_setup
    path = str(tmp_path / "setup.json")
    save_setup(path, "BLE LE-1M", {"dp_gain": 1.02}, seed=7, noise=False)
    doc = load_setup(path)
    assert doc["preset"] == "BLE LE-1M" and doc["seed"] == 7
    rep = run_setup(doc)
    assert rep["metrics"].get("mask") in ("PASS", "FAIL")


def test_setup_rejects_garbage(tmp_path):
    from polartx.guiutil import load_setup, save_setup
    p = tmp_path / "x.json"
    p.write_text("{}")
    with pytest.raises(ValueError):
        load_setup(str(p))
    with pytest.raises(ValueError):
        save_setup(str(tmp_path / "y.json"), "BLE LE-1M",
                   {"bad": object()})
