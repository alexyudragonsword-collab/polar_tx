"""Example 22: is the wideband chain converged on its default os-4 grid?
(polar_tx#22, array_tx ROADMAP E1)

The tolerances are fixed in TOL_DB before anything runs: a reading
converges when it is within TOL of the same reading at os 16, and os 16
is itself converged when os 32 is within TOL of it.  ACLR 0.3 dB and EVM
0.1 dB are array_tx's; 0.3 dB on a mask margin is ours (a margin is a power
ratio like ACLR).  Per-tone EVM and the full-span mask margin were added
after the first full run showed the hold's droop and the span the os-4
grid cannot see; their tolerances are their class's, not fitted to data.

"oversampling" used to mean two things in these presets at once: the
simulation grid and the DTC / DPA update clock.  They are separate now
(``update_oversampling``, ``ChainConfig.f_clock``), so every case is read
two ways:

  T  clock on the grid (``update_oversampling=None``): raising os clocks
     the DTC and DPA 2/4/8x faster -- array_tx's protocol, and how the
     presets behaved before.  A change here is a different chip.
  P  the preset's own hardware (clock at 4x) on a finer grid: the digital
     chain runs at the clock and its output is zero-order held onto the
     grid, jitter per update, LO noise on the grid.  Only the grid
     changes.  Checked below against an independent construction (the
     os-4 run, repeated and advanced by the hold's group delay): equal to
     1e-12 with the noise off.

Measured: ACLR1 / ACLR2 (worst side, 0.9 BW measurement bandwidth), the
stylized template's out-of-band margin over the span the os-4 grid sees
("mask") and over the whole grid ("mask_full": the WiFi templates run to
5 BW, past the hold's first image at 4 BW), scalar and per-tone EVM.
Then the documented rows (README, ex04, ex07, ex11, ex19/20) read the way
the report reads them, os 4 against os 16 with the clock pinned: what the
documented numbers would do on a converged grid.
Results go to stdout and examples/out/ex22_grid.json.
"""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np

from polartx.chain import advance
from polartx.dpa.dpa import DPAConfig
from polartx.grid import grid_verdict
from polartx.metrics.aclr_ext import aclr_multi
from polartx.metrics.masks import default_mask_spec
from polartx.metrics.sem import check_sem
from polartx.presets import (bench_wifi6_polar_benbassat20,
                             bench_wifi7_polar_degani24, nr_dtc, wifi_dtc,
                             wifi_outphasing, wifi_rfdac)
from polartx.vendor.padpd.metrics import evm_of_signal, psd

# ------------------------------------------------- written before the run
TOL_DB = {"aclr1": 0.3, "aclr2": 0.3, "mask": 0.3, "mask_full": 0.3,
          "evm": 0.1, "evm_tone": 0.1}
OS = (4, 8, 16, 32)
BASE_OS, REF_OS, CHECK_OS = 4, 16, 32
N_SYMBOLS = 8
SEED_WF, SEED_RUN = 1, 3

OUT = os.path.join(os.path.dirname(__file__), "out")
warnings.simplefilter("ignore", UserWarning)

#: the PA of the issue's reproduction (array_tx's element PA)
RAPP = DPAConfig(n_bits=10, n_thermo=6, amam=("rapp", 2.0, 1.0),
                 ampm_deg_poly=(0.0, 0.0, 8.0))

CASES = {   # name: (preset factory, bandwidth); DPD off/on x noise off/on
    "NR FR2 400 MHz 64-QAM": (lambda **kw: nr_dtc(bw=400e6, **kw), 400e6),
    "NR FR2 400 MHz, Rapp PA (the issue's)": (
        lambda **kw: nr_dtc(bw=400e6, dpa=RAPP, **kw), 400e6),
    "WiFi 320 MHz 1024-QAM": (
        lambda **kw: wifi_dtc(bw=320e6, qam=1024, **kw), 320e6),
    "WiFi 320 MHz 4096-QAM": (
        lambda **kw: wifi_dtc(bw=320e6, qam=4096, **kw), 320e6),
}
ANCHORS = {  # the documented rows' presets: view P, defaults (DPD off, noise on)
    "WiFi 80 MHz 1024-QAM": (lambda **kw: wifi_dtc(bw=80e6, **kw), 80e6),
    "WiFi 160 MHz 1024-QAM": (lambda **kw: wifi_dtc(bw=160e6, **kw), 160e6),
    "NR FR1 100 MHz 256-QAM": (lambda **kw: nr_dtc(bw=100e6, **kw), 100e6),
    "NR FR2 200 MHz 64-QAM": (lambda **kw: nr_dtc(bw=200e6, **kw), 200e6),
}
OTHER = {    # one wideband point each, conclusions only
    "outphasing WiFi 320 MHz 1024-QAM": (
        lambda **kw: wifi_outphasing(bw=320e6, qam=1024, **kw), 320e6),
    "RF-DAC WiFi 320 MHz 1024-QAM": (
        lambda **kw: wifi_rfdac(bw=320e6, qam=1024, **kw), 320e6),
}


def _bench(factory, bw, qam):
    """A preset with no grid knob (the benchmarks): its chain with the
    clock pinned at 4x, and its burst on an os-k grid."""
    from polartx.presets import burst_on_grid
    from polartx.waveforms.ofdm import wifi_waveform

    def make(oversampling=4):
        p = factory()
        p.tx.cfg.f_clock = BASE_OS * bw
        wf = burst_on_grid(lambda os_: wifi_waveform(bw, qam, n_symbols=8,
                                                     oversampling=os_, seed=0),
                           oversampling, BASE_OS)
        return p.tx, wf
    return make


def _preset(factory):
    def make(oversampling=4):
        p = factory(oversampling=oversampling)
        return p.tx, p.make_waveform(n_symbols=8, seed=0)
    return make


#: the documented rows (README / docs/index.html / ex04 / ex11 / ex19-20),
#: read the way the report reads them (GUI seeds: burst 0, run 1, noise on)
INVENTORY = {
    "WiFi 80 MHz (README, GUI)": _preset(lambda **kw: wifi_dtc(bw=80e6, **kw)),
    "WiFi 160 MHz (README, GUI default, ex19/20)": _preset(
        lambda **kw: wifi_dtc(bw=160e6, **kw)),
    "WiFi 7 320 MHz 4096-QAM (README)": _preset(
        lambda **kw: wifi_dtc(bw=320e6, qam=4096, **kw)),
    "NR FR1 100 MHz (README, ex07)": _preset(lambda **kw: nr_dtc(bw=100e6, **kw)),
    "NR FR2 200 MHz (README, ex07)": _preset(lambda **kw: nr_dtc(bw=200e6, **kw)),
    "WiFi 160 MHz, 0.2 ns skew (ex04)": _preset(
        lambda **kw: wifi_dtc(bw=160e6, env_skew_s=0.2e-9, **kw)),
    "Ben Bassat'20 raw (ex11)": _bench(
        lambda: bench_wifi6_polar_benbassat20(dpd=False), 160e6, 1024),
    "Ben Bassat'20 DPD (ex11)": _bench(
        lambda: bench_wifi6_polar_benbassat20(dpd=True), 160e6, 1024),
    "Degani'24 (ex11)": _bench(bench_wifi7_polar_degani24, 320e6, 4096),
}


def report_readings(tx, wf, span_hz):
    """The report's own numbers (PolarResult.aclr: the vendored ACLR,
    resolution held across grids; scalar and per-tone EVM; the mask
    margin within the os-4 span and over the whole grid)."""
    r = tx.run(wf, noise=True, seed=1)
    f, p = r.psd(nfft=8192 * r.grid_hold)
    spec = default_mask_spec(wf)
    keep = np.abs(f) <= span_hz
    return {"aclr_upper": float(r.aclr()["upper_dbc"]),
            "evm": float(r.evm().db),
            "evm_tone": float(r.evm(equalize="per_tone").db),
            "mask": float(check_sem(f[keep], p[keep], spec,
                                    channel_bw_hz=wf.bw)["oob_margin_db"]),
            "mask_full": float(check_sem(f, p, spec,
                                         channel_bw_hz=wf.bw)["oob_margin_db"])}


def inventory():
    print("\nDocumented rows, the report's readings, clock pinned at 4x: os 4 -> os 16")
    out = {}
    for name, make in INVENTORY.items():
        tx4, wf4 = make(4)
        span = 0.98 * wf4.fs / 2
        r4 = report_readings(tx4, wf4, span)
        tx16, wf16 = make(REF_OS)
        r16 = report_readings(tx16, wf16, span)
        out[name] = {"4": r4, "16": r16}
        print(f"  {name}")
        print("    " + "   ".join(f"{k} {r4[k]:.1f}->{r16[k]:.1f} ({r4[k] - r16[k]:+.2f})"
                                 for k in r4))
    return out


# ------------------------------------------------------------ the metrics
def metrics(y, fs, wf, bw, span_hz, k):
    """Every reading of one output on the os-``k`` grid (the FFT sizes
    scale with it, so the resolution bandwidth is the same on every grid).
    ACLR2 is None where the grid cannot hold its measurement band (WiFi at
    os 4: fs = 4 BW < 4.9 BW)."""
    out = {}
    offs = (1, 2) if fs >= 2 * (2 * bw + 0.45 * bw) else (1,)
    a = aclr_multi(y, fs, bw, offsets=offs, nfft=1024 * k)
    out["aclr1"] = max(a["aclr1_lower_dbc"], a["aclr1_upper_dbc"])
    out["aclr2"] = (max(a["aclr2_lower_dbc"], a["aclr2_upper_dbc"])
                    if 2 in offs else None)
    f, p = psd(y, fs, nfft=2048 * k)
    spec = default_mask_spec(wf)
    keep = np.abs(f) <= span_hz
    out["mask"] = float(check_sem(f[keep], p[keep], spec,
                                  channel_bw_hz=bw)["oob_margin_db"])
    out["mask_full"] = float(check_sem(f, p, spec,
                                       channel_bw_hz=bw)["oob_margin_db"])
    ref = wf.require_ofdm_ref()
    out["evm"] = float(evm_of_signal(y, ref).db)
    out["evm_tone"] = float(evm_of_signal(y, ref, equalize="per_tone").db)
    return out


def series(make, bw, *, view, noise, dpd):
    """One reading per grid.  View T ties the update clock to the grid,
    view P keeps the preset's clock (4x) and refines the grid only."""
    res = {}
    span = None
    for k in OS:
        kw = {"oversampling": k}
        if view == "T":
            kw["update_oversampling"] = None
        if dpd is not None:
            kw["dpd"] = dpd
        p = make(**kw)
        wf = p.make_waveform(n_symbols=N_SYMBOLS, seed=SEED_WF)
        assert abs(wf.fs - p.fs_bb) < 1.0, (wf.fs, p.fs_bb)
        if span is None:
            span = 0.98 * wf.fs / 2           # what the os-4 grid can see
        y = p.tx.run(wf, noise=noise, seed=SEED_RUN).y
        res[k] = metrics(y, wf.fs, wf, bw, span, k)
    return res


def held_series(make, bw):
    """View P for the chains the clock / grid split is not modeled for
    (outphasing, RF-DAC), noise off: the os-4 run zero-order held by hand,
    advanced by the hold's group delay."""
    p4 = make()
    wf4 = p4.make_waveform(n_symbols=N_SYMBOLS, seed=SEED_WF)
    clean = p4.tx.run(wf4, noise=False, seed=SEED_RUN).y
    res = {}
    for k in OS:
        m = k // BASE_OS
        wf = make(oversampling=k).make_waveform(n_symbols=N_SYMBOLS,
                                                seed=SEED_WF)
        y = advance(np.repeat(clean, m), (m - 1) / 2.0)
        res[k] = metrics(y, wf.fs, wf, bw, 0.98 * wf4.fs / 2, k)
    return res


def check_against_hand_held(make):
    """The library's view P equals the hand-made one (noise off)."""
    p4 = make()
    y4 = p4.tx.run(p4.make_waveform(n_symbols=N_SYMBOLS, seed=SEED_WF),
                   noise=False, seed=SEED_RUN).y
    p16 = make(oversampling=REF_OS)
    y16 = p16.tx.run(p16.make_waveform(n_symbols=N_SYMBOLS, seed=SEED_WF),
                     noise=False, seed=SEED_RUN).y
    m = REF_OS // BASE_OS
    ref = advance(np.repeat(y4, m), (m - 1) / 2.0)
    err = float(np.max(np.abs(y16 - ref)) / np.max(np.abs(ref)))
    assert err < 1e-12, err
    return err


def verdict(ser):
    """os-4 minus os-16, os-32 minus os-16, each against TOL_DB
    (``polartx.grid.grid_verdict``)."""
    return grid_verdict(ser, TOL_DB, base=BASE_OS, ref=REF_OS, check=CHECK_OS)


def fmt(v):
    return "    n/a" if v is None else f"{v:7.2f}"


def table(label, ser):
    v = verdict(ser)
    print(f"\n{label}")
    print("          " + "".join(f"   os {k:<3d}" for k in OS)
          + "  os4-os16 os32-os16  verdict")
    for key in TOL_DB:
        row = "".join(f"  {fmt(ser[k][key])}" for k in OS)
        d = v[key]
        ok = ("n/a" if d["base_ok"] is None else
              ("os4 OK" if d["base_ok"] else "os4 FAIL")) + (
              "" if d["ref_ok"] in (None, True) else " / os16 NOT converged")
        print(f"  {key:8s}{row}   {fmt(d['d_base'])}   {fmt(d['d_check'])}   {ok}")
    return v


def record(results, label, ser):
    results["cases"][label] = {"series": {str(k): s for k, s in ser.items()},
                               "verdict": table(label, ser)}


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    results = {"tolerance_db": TOL_DB, "ref_os": REF_OS, "check_os": CHECK_OS,
               "cases": {}}
    err = check_against_hand_held(CASES["WiFi 320 MHz 1024-QAM"][0])
    print(f"view P vs the hand-held os-4 run (noise off): {err:.1e}")
    results["hand_held_check"] = err
    for name, (make, bw) in CASES.items():
        for dpd in (False, True):
            for noise in (False, True):
                for view in ("T", "P"):
                    label = (f"{name} | DPD {'on' if dpd else 'off'} | "
                             f"noise {'on' if noise else 'off'} | view {view}")
                    record(results, label, series(make, bw, view=view,
                                                  noise=noise, dpd=dpd))
    for name, (make, bw) in ANCHORS.items():
        record(results, f"{name} | defaults | view P",
               series(make, bw, view="P", noise=True, dpd=None))
    for name, (make, bw) in OTHER.items():
        for noise in (False, True):
            record(results, f"{name} | noise {'on' if noise else 'off'} | view T",
                   series(make, bw, view="T", noise=noise, dpd=None))
        record(results, f"{name} | noise off | view P (held by hand)",
               held_series(make, bw))
    results["inventory"] = inventory()
    with open(os.path.join(OUT, "ex22_grid.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\n{time.time() - t0:.0f} s; written {os.path.join(OUT, 'ex22_grid.json')}")
