"""The front-end contract (guispec) held to the library it describes.

Three things a hand-written form got wrong, each now a test:
defaults typed into a form instead of read from the preset (the RF-DAC
ran at 11 bits instead of its 12); knobs offered where they act on nothing
(skew on outphasing and on BLE); and a picture that is not the data the
numbers came from.
"""
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from polartx.guispec import (GROUPS, decimate_peak, knob_specs, metric_info,
                             overrides_from_ui)
from polartx.guiutil import PRESETS, build_preset, run_chain_report

CHAINS = [n for n in PRESETS if not n.startswith("Bench:")]


def _run(name, noise=False, **ov):
    """Output samples; a warning that a setting is ignored fails here, any
    other warning (an engine leaving its domain) is the knob's honest
    consequence and is allowed."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p = build_preset(name, **ov)
        y = p.tx.run(p.make_waveform(), noise=noise, seed=1).y
    ignored = [str(x.message) for x in w if "ignored" in str(x.message)]
    assert not ignored, (name, ov, ignored)
    return y


def _alt(k):
    """A legal value different from the default."""
    if k.kind == "bool":
        return not k.default
    if k.kind == "choice":
        return next(c[0] for c in k.choices if c[0] != k.default)
    if k.default == "off":
        return round((k.lo + k.hi) / 2 / k.step) * k.step
    up = k.default + 2 * k.step
    v = up if up <= k.hi else k.default - 2 * k.step
    return int(v) if k.kind == "int" else round(v, 9)


# ----------------------------------------------------------- the knobs
def test_every_chain_offers_knobs_and_benchmarks_none():
    for name in PRESETS:
        specs = knob_specs(name)
        if name.startswith("Bench:"):
            assert specs == [], name
        else:
            assert specs, name
            ids = [k.id for k in specs]
            assert len(ids) == len(set(ids)), name
            assert {k.group for k in specs} <= set(GROUPS), name
    with pytest.raises(ValueError):
        knob_specs("no such preset")


@pytest.mark.parametrize("name", CHAINS)
def test_the_form_defaults_are_the_preset_defaults(name):
    """Every knob at its displayed default must build the very chain the
    preset builds with no overrides — bit for bit.  A default typed into a
    form (or a wrong unit scale) breaks this; the RF-DAC's 12 bits, offered
    as 11 by the old form, is the case it was written for."""
    specs = knob_specs(name)
    for k in specs:
        if k.default == "off":
            continue
        assert k.lo <= k.default <= k.hi if k.lo is not None else True, k
    ui = {k.id: k.default for k in specs}
    y_form = _run(name, **overrides_from_ui(name, ui))
    y_preset = _run(name)
    assert np.array_equal(y_form, y_preset), name


@pytest.mark.parametrize("name", CHAINS)
def test_every_knob_moves_the_output(name):
    """A knob the form offers must act on this chain, measured on the
    output samples.  Knobs that act only through the noise (jitter; the
    loop bandwidth with the two points matched) are compared with noise
    on, same seed."""
    for k in knob_specs(name):
        noise = k.noise_only
        y0 = _run(name, noise=noise)
        y1 = _run(name, noise=noise,
                  **overrides_from_ui(name, {k.id: _alt(k)}))
        assert y1.shape != y0.shape or not np.allclose(y0, y1), (name, k.id)


def test_unit_conversion_round_trips_and_off_means_none():
    k = {s.id: s for s in knob_specs("WiFi 160 MHz")}
    assert k["skew_ns"].to_override(0.2) == pytest.approx(0.2e-9)
    assert k["jitter_fs"].default == 50.0                  # wifi_dtc: 5e-14 s
    assert k["cfr_db"].default == 8.5
    assert overrides_from_ui("WiFi 160 MHz", {"cfr_db": "off"}) == \
        {"cfr_papr_db": None}
    lte = {s.id: s for s in knob_specs("LTE 20 MHz")}
    assert lte["cfr_db"].default == "off"                  # lte20: None
    assert lte["dp_err_pct"].default == 0.0
    assert lte["dp_err_pct"].to_override(5.0) == pytest.approx(1.05)
    assert lte["loop_bw_khz"].default == 1000.0
    rf = {s.id: s for s in knob_specs("WiFi 160 MHz (RF-DAC)")}
    assert rf["dac_bits"].default == 12 and "dtc_bits" not in rf
    # ids a preset does not offer are dropped, not passed through
    assert overrides_from_ui("WiFi 160 MHz (RF-DAC)",
                             {"dtc_bits": 9, "skew_ns": 1.0}) == {}
    with pytest.raises(ValueError):
        overrides_from_ui("LTE 20 MHz", {"mode": "fast"})


def test_the_legacy_knob_table_is_a_subset_of_the_specs():
    """Qt and Streamlit still send keyword-keyed values through
    guiutil.chain_overrides until they are retired; whatever that table
    lets through must be something the spec offers."""
    from polartx.guiutil import honoured_knobs
    for name in PRESETS:
        assert set(honoured_knobs(name)) <= {k.kw for k in knob_specs(name)}


# --------------------------------------------------------- the metrics
def test_every_reported_metric_is_described():
    """A metric that reaches the cards must have a label, a unit and a
    direction; a new key with no description fails here, not on screen."""
    import matplotlib.pyplot as plt
    for name in PRESETS:
        rep = run_chain_report(name, seed=1)
        for key in rep["metrics"]:
            info = metric_info(key)
            assert info["zh"] and info["en"], key
            assert info["better"] in ("lower", "higher", None)
        plt.close(rep["fig"])
    assert metric_info("in-band +-4 MHz [dBc]")["unit"] == "dBc"
    with pytest.raises(KeyError):
        metric_info("made-up metric")


# ----------------------------------------------- conventions & warnings
def test_reports_state_their_conventions():
    rep = run_chain_report("WiFi 160 MHz", seed=1)
    conv = {c["id"]: c for c in rep["conventions"]}
    assert "scalar" in conv["evm_eq"]["en"]
    assert rep["result"].evm_equalize_default == "scalar"
    assert "stylized" in conv["mask"]["en"]
    lte = {c["id"]: c for c in run_chain_report("LTE 20 MHz", seed=1)
           ["conventions"]}
    assert "response" in lte["engine"]["en"]                # linearized
    assert "engine" not in conv                             # DTC: no ADPLL


def test_a_run_time_warning_reaches_the_report():
    """Running LTE-20 on the event engine leaves its validity domain; the
    chain warns, and that warning used to die inside the report layer."""
    rep = run_chain_report("LTE 20 MHz", seed=1, mode="event")
    assert any("validity domain" in m for m in rep["warnings"])
    assert run_chain_report("LTE 20 MHz", seed=1)["warnings"] == []


# ------------------------------------------------------------ the plots
def test_the_picture_is_drawn_from_the_plot_data():
    """The PNG and the phone's charts draw the same arrays: the matplotlib
    lines carry exactly the plot series."""
    import matplotlib.pyplot as plt
    for name in ("WiFi 160 MHz", "BLE LE-1M", "BT EDR3 8DPSK"):
        rep = run_chain_report(name, seed=1)
        assert len(rep["plots"]) == len(rep["fig"].axes) == 2
        for ax, pl in zip(rep["fig"].axes, rep["plots"]):
            lines = ax.get_lines()
            assert len(lines) == len(pl["series"])
            for ln, s in zip(lines, pl["series"]):
                assert np.array_equal(ln.get_xdata(), s["x"])
                assert np.array_equal(ln.get_ydata(), s["y"])
        spec = rep["plots"][0]
        assert len(spec["series"][0]["x"]) <= 2048
        plt.close(rep["fig"])


def test_display_decimation_keeps_the_peak():
    """Max-hold: a one-bin spur survives 4x thinning at full height."""
    x = np.arange(8192.0)
    y = np.full(8192, -80.0)
    y[5001] = -20.0
    xs, ys = decimate_peak(x, y)
    assert len(xs) == 2048 and ys.max() == -20.0 and 5001.0 in xs
