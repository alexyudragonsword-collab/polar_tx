"""BT EDR DPSK payloads through the narrowband polar chain (DEVM)."""
import numpy as np
import pytest

from polartx.metrics.dpsk import devm
from polartx.presets import bt_edr_adpll
from polartx.waveforms.edr import edr_dpsk


def test_metric_floor_on_ideal_burst():
    """Matched-filter DEVM floor (SRRC truncation only) is far below the
    impairments under study."""
    wf = edr_dpsk(600, 32e6, mode="8dpsk", seed=2)
    assert devm(wf.x, wf)["devm_pct"] < 0.5


def test_envelope_path_is_exercised():
    """EDR is not constant-envelope: the DPA sees a wide code range."""
    p = bt_edr_adpll("8dpsk")
    res = p.tx.run(p.make_waveform(n_syms=400, seed=2), noise=False)
    assert np.unique(res.env_code).size > 50
    papr = 10 * np.log10(np.max(res.env_cmd ** 2) /
                         np.mean(res.env_cmd ** 2))
    assert 2.0 < papr < 4.5


@pytest.mark.parametrize("mode,limit_pct", [("pi4dqpsk", 20.0),
                                            ("8dpsk", 13.0)])
def test_devm_within_spec_limits(mode, limit_pct):
    p = bt_edr_adpll(mode)
    wf = p.make_waveform(n_syms=600, seed=2)
    res = p.tx.run(wf, noise=True, seed=3)
    d = res.evm()
    assert d["devm_pct"] < 0.5 * limit_pct     # comfortable margin
    assert res.check_mask()[0]


def test_short_burst_refuses_instead_of_returning_nan():
    """Regression: devm trims `span` symbols off each end for the pulse
    edge transient. A burst shorter than both trims left an EMPTY slice,
    whose mean is NaN — so the metric silently returned nan% and carried
    it into reports and the GUI as "nan%" with nothing naming the cause.
    Refusing is the honest failure; NaN propagates."""
    from polartx.presets import bench_edge_polar_staszewski05
    p = bench_edge_polar_staszewski05()
    span = p.make_waveform(64, seed=1).meta["span"]

    with pytest.raises(ValueError, match="too short"):
        wf = p.make_waveform(2 * span, seed=1)
        p.tx.run(wf, noise=True, seed=1).evm()

    # just past the floor it scores normally, and finitely
    wf = p.make_waveform(2 * span + 8, seed=1)
    d = p.tx.run(wf, noise=True, seed=1).evm()["devm_pct"]
    assert np.isfinite(d) and 0.0 < d < 10.0
