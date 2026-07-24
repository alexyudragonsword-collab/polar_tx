"""Power-detector skew search and the phase-interp shape trade."""
import numpy as np

from polartx.cal.skew import estimate_skew_by_acp
from polartx.chain import ChainConfig
from polartx.metrics.ble_metrics import bt_acp
from polartx.presets import bt_edr_adpll, wifi_dtc


def test_acp_search_recovers_skew():
    """No waveform-domain observation, only band power: grid search +
    parabolic refinement lands within ~0.4 ns on a 2.3 ns skew."""
    p = wifi_dtc(bw=160e6, env_skew_s=2.3e-9)
    wf = p.make_waveform(n_symbols=4, seed=2)
    est = estimate_skew_by_acp(p.tx, wf, span_s=5e-9, n_grid=7)
    assert abs(est["skew_s"] - 2.3e-9) < 0.4e-9
    assert est["acp_db"].argmin() not in (0, len(est["acp_db"]) - 1)


def test_phase_interp_shapes_both_comply():
    """Both transition shapes respect the slew bound; at a FIXED bound
    the wider smoothstep window costs more trajectory EVM than its
    continuous derivative saves (documented finding, ex08)."""
    wf = bt_edr_adpll("8dpsk").make_waveform(n_syms=400, seed=2)
    out = {}
    for shape in ("linear", "smooth"):
        ch = ChainConfig(env_floor=0.05, phase_slew_max_hz=2e6,
                         phase_interp=shape)
        q = bt_edr_adpll("8dpsk", mode="event", dp_range_hz=2e6, chain=ch)
        r = q.tx.run(wf, noise=True, seed=3)
        assert r.info["phasemod"]["dp_required_range_hz"] <= 2e6 * 1.01
        assert r.info["phasemod"]["dp_clip_frac"] == 0.0
        acp = bt_acp(r.y, r.fs)
        out[shape] = (r.info["split"]["mod_evm_db"],
                      max(acp["acp+2MHz_dbc"], acp["acp-2MHz_dbc"]))
    # linear's shorter windows -> smaller trajectory modification
    assert out["linear"][0] < out["smooth"][0]
    assert out["linear"][1] < out["smooth"][1] + 1.0