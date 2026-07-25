"""Literature-class benchmark presets stay in their published classes."""
import polartx
from polartx.guiutil import PRESETS, build_preset, run_chain_report
from polartx.metrics.aclr_ext import aclr_multi
from polartx.presets import (bench_edge_polar_staszewski05,
                             bench_lte20_polar_madoglio14,
                             bench_wifi6_polar_benbassat20,
                             bench_wifi7_polar_degani24,
                             bench_wifi11n_polar)

BENCH_PRESETS = ["Bench: Staszewski'05 EDGE", "Bench: Madoglio'14 LTE-20",
                 "Bench: BenBassat'20 WiFi6", "Bench: Degani'24 WiFi7",
                 "Bench: 802.11n polar (~2010)"]

#: benchmarks that cannot be a registry entry: they do not return the
#: single-PolarTX TxPreset shape the chain report layer runs.
_NON_TXPRESET_BENCHES = {"bench_wifi7_mlo_fir_borokhovich26"}  # FIRTxPreset


def test_benchmarks_are_registered_presets():
    for name in BENCH_PRESETS:
        assert name in PRESETS
    assert len(PRESETS) == 10 + len(BENCH_PRESETS)   # 10 standard chains


def test_every_benchmark_is_reachable_from_the_gui():
    """Guards the failure mode where a new benchmark lands in presets.py
    but never reaches the GUI: every exported bench_* factory must either
    be a registry entry or have its own guiutil entry point."""
    from polartx import guiutil
    exported = {n for n in polartx.__all__ if n.startswith("bench_")}
    registered = set()
    for name in PRESETS:
        if name.startswith("Bench:"):
            registered.add(name)
    # one registry entry per TxPreset-shaped benchmark
    assert len(registered) == len(exported - _NON_TXPRESET_BENCHES)
    # and the odd-shaped one is reachable through its dedicated report
    assert hasattr(guiutil, "run_fir_report")


def test_benchmarks_exported_top_level():
    for fn in ("bench_edge_polar_staszewski05",
               "bench_lte20_polar_madoglio14", "bench_wifi11n_polar"):
        assert fn in polartx.__all__ and hasattr(polartx, fn)


def test_benchmarks_run_through_report_layer():
    """The GUI/report path builds and scores each benchmark, dispatching
    the right burst-length kwarg via signature inspection (EDGE uses
    n_syms, the others n_symbols)."""
    for name in BENCH_PRESETS:
        rep = run_chain_report(name, seed=1, noise=True, n_units=200)
        assert rep["fig"] is not None
        m = rep["metrics"]
        assert ("EVM [dB]" in m) or ("DEVM [%]" in m)


def test_benchmarks_ignore_overrides():
    """Benchmarks fix their published-class parameters — a stray
    override must not perturb them."""
    a = build_preset("Bench: Madoglio'14 LTE-20")
    b = build_preset("Bench: Madoglio'14 LTE-20", dp_gain=1.5, n_bits=6)
    wf = a.make_waveform(n_symbols=6, seed=0)
    ea = a.tx.run(wf, noise=False).evm().db
    eb = b.tx.run(b.make_waveform(n_symbols=6, seed=0), noise=False).evm().db
    assert abs(ea - eb) < 1e-9


def test_edge_polar_class():
    p = bench_edge_polar_staszewski05()
    r = p.tx.run(p.make_waveform(400, seed=1), noise=True, seed=1)
    d = r.evm()["devm_pct"]
    assert 1.0 < d < 3.5           # published class 2-3%, spec 9%


def test_lte20_polar_class():
    p = bench_lte20_polar_madoglio14()
    r = p.tx.run(p.make_waveform(n_symbols=8, seed=0), noise=True, seed=1)
    assert -34.0 < r.evm().db < -28.0        # class ~-30 dB
    a = aclr_multi(r.y, r.fs, 20e6, offsets=(1,))
    assert max(a["aclr1_lower_dbc"], a["aclr1_upper_dbc"]) < -33.0


def test_wifi6_polar_benbassat_class():
    """ISSCC/JSSC 2020 Intel Wi-Fi 6: 160 MHz 1024-QAM, raw EVM ~-29 dB,
    -40 dB class with DPD."""
    raw = bench_wifi6_polar_benbassat20(dpd=False)
    r0 = raw.tx.run(raw.make_waveform(n_symbols=6, seed=0), noise=True,
                    seed=1)
    assert -32.0 < r0.evm().db < -26.0       # published raw ~-29/-30.5
    p = bench_wifi6_polar_benbassat20()      # DPD on (default)
    r1 = p.tx.run(p.make_waveform(n_symbols=6, seed=0), noise=True, seed=1)
    assert r1.evm().db < -36.0               # -40 dB class (LO-limited @160)
    assert r1.evm().db < r0.evm().db - 6.0   # DPD clearly helps


def test_wifi7_polar_degani_class():
    """RFIC 2024 Intel Wi-Fi 7: 320 MHz 4096-QAM (MCS13), -38 dB class
    with DPD, watt-level SCPA (34.7% peak efficiency)."""
    p = bench_wifi7_polar_degani24()             # DPD on (default)
    wf = p.make_waveform(n_symbols=6, seed=0)
    r = p.tx.run(wf, noise=True, seed=1)
    assert r.evm().db < -35.0                    # -38 dB class (LO-limited)
    eff = r.avg_efficiency(p.tx.dpa)
    assert 0.12 < eff["eta_avg"] < 0.30          # backoff efficiency
    raw = bench_wifi7_polar_degani24(dpd=False)
    r0 = raw.tx.run(wf, noise=True, seed=1)
    assert r0.evm().db > r.evm().db + 6.0         # DPD clearly helps


def test_wifi11n_polar_still_available():
    """Historical anchor, importable but no longer in the GUI registry."""
    p = bench_wifi11n_polar()
    r = p.tx.run(p.make_waveform(n_symbols=6, seed=0), noise=True, seed=1)
    assert -32.0 < r.evm().db < -25.0        # class ~-28, spec -25
