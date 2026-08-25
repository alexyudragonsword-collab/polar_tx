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
