"""The Android bridge, exercised without Android.

The whole contract is JSON in, JSON out, so every method the phone can
reach is reachable from pytest in milliseconds.  That is the point of the
one-function design: it collapses the phone-only work to "does the UI
render this", which a person can check in a minute.
"""
import json

import numpy as np
import pytest

from polartx import appbridge


def _ok(method, **args):
    r = json.loads(appbridge.call(method, json.dumps(args)))
    assert r["ok"], r.get("error", "") + "\n" + r.get("traceback", "")
    return r["result"]


def _err(method, **args):
    r = json.loads(appbridge.call(method, json.dumps(args)))
    assert not r["ok"], f"expected {method} to fail, got {r}"
    return r


# ------------------------------------------------------------- envelope

def test_every_method_round_trips_as_json():
    """Every registered method returns something json.dumps already
    accepted — the envelope is built by call() itself, so a method that
    returns a numpy scalar or a bare NaN would have raised there."""
    cheap = {"version": {}, "list_presets": {},
             "chain": {"preset": "BLE LE-1M"},
             "sweep": {"preset": "BLE LE-1M", "knob": "dp_err_pct",
                       "points": [0.0, 2.0]},
             "compare": {},
             "calibrate": {"kind": "dtc", "iterations": 1},
             "measured": {},
             "fir": {"n_symbols": 2},
             "selector": {}, "combiner": {},
             "montecarlo": {"n_chips": 3}}
    assert set(cheap) == set(appbridge._METHODS), (
        "a method was added without a case here; every entry in _METHODS "
        "must be exercised or it ships untested")
    for m, args in cheap.items():
        _ok(m, **args)


def test_unknown_method_comes_back_in_band():
    """Not raised across the FFI: a Python exception reaching Kotlin arrives
    as a PyException the WebView cannot render."""
    r = _err("no_such_method")
    assert "no such method" in r["error"]
    assert "known:" in r["error"], "the error should name what IS available"
    assert r["traceback"]


def test_bad_arguments_come_back_in_band():
    r = _err("chain", preset="BLE LE-1M", nonexistent_kwarg=1)
    assert "TypeError" in r["error"]


def test_non_object_arguments_are_refused():
    r = json.loads(appbridge.call("version", "[1, 2, 3]"))
    assert not r["ok"] and "JSON object" in r["error"]


# -------------------------------------------------------------- _clean

def test_clean_makes_numpy_and_nan_json_safe():
    """json.dumps raises on np.float64 and emits a bare NaN token for NaN,
    which is invalid JSON and fails inside the WebView rather than here —
    far from anything that points at the cause."""
    out = appbridge._clean({"a": np.float64(1.5), "b": np.int64(3),
                            "c": np.bool_(True), "d": float("nan"),
                            "e": np.array([1.0, 2.0])})
    assert out == {"a": 1.5, "b": 3, "c": True, "d": None, "e": [1.0, 2.0]}
    text = json.dumps(out)                      # would raise on the raw dict
    assert "NaN" not in text


# ------------------------------------------------------------- methods

def test_version_names_the_library_and_runtime():
    v = _ok("version")
    assert v["lib"] and v["python"] and v["numpy"]


def test_list_presets_splits_benchmarks_out():
    """The page groups them separately, and a bench_* factory that never
    reached guiutil.PRESETS would show up here as an empty group."""
    from polartx.guiutil import PRESETS
    p = _ok("list_presets")
    assert p["benchmarks"], "no benchmarks reached the phone"
    assert all(n.startswith("Bench:") for n in p["benchmarks"])
    assert not any(n.startswith("Bench:") for n in p["presets"])
    assert len(p["presets"]) + len(p["benchmarks"]) == len(PRESETS)


def test_chain_returns_metrics_and_a_rendered_plot():
    r = _ok("chain", preset="BLE LE-1M")
    assert r["metrics"], "no metrics"
    assert len(r["png"]) > 1000, "the figure did not render"
    import base64
    assert base64.b64decode(r["png"])[:8] == b"\x89PNG\r\n\x1a\n"


def test_chain_skew_is_in_nanoseconds_and_actually_bites():
    """Two things at once: the ns->s conversion is real (not a knob wired to
    nothing — this repository has shipped one of those), and the damage
    lands where the physics says, on the OFDM chain and not on BLE.
    """
    key = "mask OOB margin [dB]"
    clean = _ok("chain", preset="WiFi 160 MHz", env_skew_ns=0.0)
    skewed = _ok("chain", preset="WiFi 160 MHz", env_skew_ns=0.5)
    assert float(clean["metrics"][key]) > float(skewed["metrics"][key]) + 10.0

    # constant-envelope BLE has no envelope to skew: bit-identical
    a = _ok("chain", preset="BLE LE-1M", env_skew_ns=0.0)
    b = _ok("chain", preset="BLE LE-1M", env_skew_ns=5.0)
    assert a["metrics"] == b["metrics"]


def test_fir_refuses_a_notch_outside_the_represented_band():
    """The house rule: a metric with insufficient input raises rather than
    returning NaN.  Without the guard the OOC suppression averages an empty
    slice and reaches the page as a blank card with nothing pointing at the
    cause.
    """
    r = _err("fir", notch_offset_hz=5e9, n_symbols=2)
    assert "notch offset" in r["error"] and "osr" in r["error"]
    # and the same request inside the band succeeds
    got = _ok("fir", notch_offset_hz=5e8, n_symbols=2)
    assert np.isfinite(float(got["metrics"]["OOC suppression [dB]"]))


def test_selector_carries_its_breakdown_table():
    """The verdict alone is oracular; the per-contributor breakdown is what
    makes it arguable, so the phone gets it too."""
    r = _ok("selector", bw_hz=160e6)
    assert "recommendation" in r["metrics"]
    assert "adpll_two_point" in r["table"] and "dtc_open_loop" in r["table"]


def test_selector_recommends_the_two_architectures_in_the_right_regimes():
    narrow = _ok("selector", bw_hz=2e6, constant_envelope=True,
                 evm_db_max=-25.0, fout=2.44e9)
    wide = _ok("selector", bw_hz=320e6, evm_db_max=-38.0, fout=6e9)
    assert "ADPLL" in narrow["metrics"]["recommendation"]
    assert "DTC" in wide["metrics"]["recommendation"]


def test_combiner_beats_a_single_core_at_backoff():
    r = _ok("combiner", n_way=2, backoff_db=6.0)
    m = r["metrics"]
    assert m["avg eff Doherty [%]"] > m["avg eff single-core SCPA [%]"]


def test_montecarlo_summary_is_a_distribution_not_just_a_yield():
    s = _ok("montecarlo", n_chips=5)["summary"]
    for k in ("n", "mean", "std", "p95", "worst", "best", "yield"):
        assert k in s, f"summary lost {k}"
    assert s["n"] == 5


@pytest.mark.parametrize("preset", ["BLE LE-1M", "BT EDR3 8DPSK",
                                    "LTE 20 MHz", "WiFi 80 MHz",
                                    "NR FR1 100 MHz",
                                    "Bench: Degani'24 WiFi7"])
def test_a_representative_preset_from_each_family_runs_on_the_bridge(preset):
    """One per waveform kind (GFSK / DPSK / SC-FDMA / OFDM / NR / bench):
    the report layer dispatches burst length by signature inspection, and
    that dispatch is exactly what breaks when a preset family is added."""
    r = _ok("chain", preset=preset)
    assert r["metrics"] and r["png"]


def test_chain_drops_a_knob_the_preset_does_not_honour():
    """The phone shows the skew field only where it applies, and the
    bridge enforces the same table: a skew sent with the RF-DAC preset is
    dropped (the chain would warn and ignore it), and the page learns the
    default preset and the knob table from list_presets."""
    from polartx.guiutil import DEFAULT_PRESET, PRESETS
    p = _ok("list_presets")
    assert p["default"] == DEFAULT_PRESET
    assert set(p["knobs"]) == set(PRESETS)
    assert "env_skew_s" not in p["knobs"]["WiFi 160 MHz (RF-DAC)"]
    assert "env_skew_s" in p["knobs"]["WiFi 160 MHz"]
    # the report now carries the run's warnings itself, so "the chain was
    # never asked to ignore anything" is read from there
    a = _ok("chain", preset="WiFi 160 MHz (RF-DAC)", env_skew_ns=0.0)
    b = _ok("chain", preset="WiFi 160 MHz (RF-DAC)", env_skew_ns=1.0)
    assert a["metrics"] == b["metrics"]
    assert a["warnings"] == b["warnings"] == []


def test_chain_takes_knob_ids_and_returns_drawable_data():
    """What the redesigned page needs from one call: controls by knob id
    in display units, and back the numbers, the plot data they were drawn
    from, the conventions, the warnings and a description of each metric.
    png=False skips the figure; the numbers do not change."""
    from polartx.guispec import knob_specs
    p = _ok("list_presets")
    assert [k["id"] for k in p["specs"]["WiFi 160 MHz"]] == \
        [k.id for k in knob_specs("WiFi 160 MHz")]
    assert set(p["groups"]) >= {k["group"] for k in p["specs"]["LTE 20 MHz"]}

    skewed = _ok("chain", preset="WiFi 160 MHz", values={"skew_ns": 0.5},
                 png=False)
    legacy = _ok("chain", preset="WiFi 160 MHz", env_skew_ns=0.5)
    assert skewed["metrics"] == legacy["metrics"]
    assert "png" not in skewed and len(legacy["png"]) > 1000
    spec, right = skewed["plots"]
    assert spec["id"] == "spectrum" and len(spec["series"][0]["x"]) <= 2048
    assert len(right["series"][0]["x"]) == len(right["series"][0]["y"]) > 100
    assert set(skewed["metric_info"]) == set(skewed["metrics"])
    assert any(c["id"] == "evm_eq" for c in skewed["conventions"])
    # an id this preset does not offer is dropped, not misapplied
    clean = _ok("chain", preset="WiFi 160 MHz (RF-DAC)",
                values={"skew_ns": 1.0, "dtc_bits": 6}, png=False)
    base = _ok("chain", preset="WiFi 160 MHz (RF-DAC)", png=False)
    assert clean["metrics"] == base["metrics"]
    # a run-time warning arrives with the result
    ev = _ok("chain", preset="LTE 20 MHz", values={"mode": "event"}, png=False)
    assert any("validity domain" in w for w in ev["warnings"])


def test_sweep_and_compare_cross_the_bridge():
    sw = _ok("sweep", preset="WiFi 160 MHz", knob="skew_ns",
             points=[0.0, 0.3])
    assert sw["x"] == [0.0, 0.3] and sw["knob"]["unit"] == "ns"
    evm = sw["metrics"]["EVM [dB]"]
    assert evm[1] > evm[0] + 10.0                  # the skew bites
    assert sw["metrics"]["mask"] == ["PASS", "FAIL"]
    bad = _err("sweep", preset="WiFi 160 MHz (RF-DAC)", knob="skew_ns")
    assert "offers no control" in bad["error"]
    cmp = _ok("compare", values={"dtc_bits": 9})
    assert len(cmp["names"]) == 3 and set(cmp["metrics"]) == set(cmp["names"])
    assert set(cmp["keys"]) == set(cmp["metric_info"])
    assert len(cmp["plots"][0]["series"]) == 4                 # 3 spectra + mask


def test_calibration_measured_and_selector_handoff_cross_the_bridge():
    sk = _ok("calibrate", kind="skew", skew_ns=1.5)
    assert sk["error_ps"] < 10.0 and abs(sk["estimated_ns"] - 1.5) < 0.01
    tp = _ok("calibrate", kind="twopoint", gain_err_pct=4.0)
    assert tp["evm_after_db"] < tp["evm_before_db"] - 20.0
    assert "unknown calibration" in _err("calibrate", kind="guess")["error"]
    m = _ok("measured")
    assert m["metrics"]["slow-state gain [dB]"] >= 6.0          # measured 13.3
    assert any("synthetic" in c["en"] for c in m["conventions"])
    assert [p["id"] for p in m["plots"]] == ["step", "ladder"]
    sel = _ok("selector", bw_hz=160e6)
    from polartx.guiutil import PRESETS
    assert sel["workbench_preset"] in PRESETS
    assert sel["plots"][0]["xlog"] is True
