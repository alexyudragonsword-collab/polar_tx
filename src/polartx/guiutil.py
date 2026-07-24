"""Pure-computation layer behind the Streamlit GUI (testable headless).

Every function returns plain data + matplotlib figures; the GUI pages
only lay them out.  Mirrors the sibling repos' gui_core/services split.
"""
from __future__ import annotations

import numpy as np

def _registry():
    """Display name -> factory(**overrides) -> TxPreset.

    Standard chains plus the literature-class benchmarks (pllsim ex14
    convention); benchmarks ignore overrides that would break their
    published-class assumptions by fixing their own parameters."""
    from . import presets as P
    return {
        "BLE LE-1M": lambda **o: P.ble_1m_adpll(**o),
        "BLE LE-2M": lambda **o: P.ble_2m_adpll(**o),
        "BT EDR2 pi/4-DQPSK": lambda **o: P.bt_edr_adpll("pi4dqpsk", **o),
        "BT EDR3 8DPSK": lambda **o: P.bt_edr_adpll("8dpsk", **o),
        "LTE 20 MHz": lambda **o: P.lte20_adpll(**o),
        "WiFi 80 MHz": lambda **o: P.wifi_dtc(bw=80e6, **o),
        "WiFi 160 MHz": lambda **o: P.wifi_dtc(bw=160e6, **o),
        "WiFi 320 MHz": lambda **o: P.wifi_dtc(bw=320e6, **o),
        "NR FR1 100 MHz": lambda **o: P.nr_dtc(bw=100e6, **o),
        "NR FR2 200 MHz": lambda **o: P.nr_dtc(bw=200e6, **o),
        "Bench: Staszewski'05 EDGE": lambda **o: P.bench_edge_polar_staszewski05(),
        "Bench: Madoglio'14 LTE-20": lambda **o: P.bench_lte20_polar_madoglio14(),
        "Bench: BenBassat'20 WiFi6": lambda **o: P.bench_wifi6_polar_benbassat20(),
        "Bench: Degani'24 WiFi7": lambda **o: P.bench_wifi7_polar_degani24(),
    }


#: chain presets selectable in the GUIs and the headless report layer
PRESETS = list(_registry())


def build_preset(name: str, **overrides):
    """Preset registry -> TxPreset; overrides forwarded to the factory
    (benchmark presets fix their own parameters and ignore overrides)."""
    reg = _registry()
    if name not in reg:
        raise ValueError(f"unknown preset {name!r}")
    return reg[name](**overrides)


def _size_kwargs(make_waveform, n_units: int | None) -> dict:
    """Map a burst-length request onto whatever length kwarg this
    preset's make_waveform actually accepts (n_bits / n_syms /
    n_symbols) — robust to the per-preset waveform signature."""
    if n_units is None:
        return {}
    import inspect
    params = inspect.signature(make_waveform).parameters
    for kw in ("n_bits", "n_syms", "n_symbols"):
        if kw in params:
            return {kw: n_units}
    return {}


def run_chain_report(name: str, *, seed: int = 1, noise: bool = True,
                     n_units: int | None = None, **overrides) -> dict:
    """Build, run, and package metrics + figures for one chain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .metrics import check_mask
    from .metrics.masks import default_mask

    p = build_preset(name, **overrides)
    wf = p.make_waveform(**_size_kwargs(p.make_waveform, n_units))
    res = p.tx.run(wf, noise=noise, seed=seed)

    metrics = {}
    e = res.evm()
    if hasattr(e, "db"):
        metrics["EVM [dB]"] = round(e.db, 1)
        metrics["EVM [%]"] = round(e.percent, 2)
    elif "devm_pct" in e:
        metrics["DEVM [%]"] = round(e["devm_pct"], 2)
    else:
        metrics["phase EVM [%]"] = round(e["evm_pct"], 2)
    try:
        a = res.aclr()
        metrics["ACLR upper [dBc]"] = round(float(a["upper_dbc"]), 1)
    except ValueError:
        pass
    f, pdb = res.psd(nfft=8192)
    ok, margin, mask_db = check_mask(f, pdb, default_mask(wf))
    metrics["mask"] = "PASS" if ok else "FAIL"

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(f / 1e6, pdb, lw=0.7)
    ax[0].plot(f / 1e6, mask_db, "r--", lw=1.0)
    ax[0].set(xlabel="offset [MHz]", ylabel="dBr", ylim=(-90, 5),
              title=f"{name} spectrum")
    if wf.kind == "ofdm":
        from .waveforms.ofdm import demodulate_ofdm
        rx = demodulate_ofdm(res.y, wf.ofdm_ref)
        tx_s = wf.ofdm_ref.tx_symbols
        g = np.vdot(tx_s, rx) / np.vdot(tx_s, tx_s)
        pts = (rx / g).ravel()
        ax[1].plot(pts.real, pts.imag, ".", ms=1, alpha=0.4)
        ax[1].set_title("constellation")
    elif wf.kind == "dpsk":
        z = e["symbols_rx"][20:-20]
        ax[1].plot(z.real, z.imag, ".", ms=2, alpha=0.5)
        ax[1].set_title("DPSK symbols")
    else:
        y = res.y / np.abs(res.y).max()
        ax[1].plot(y.real[2000:12000], y.imag[2000:12000], lw=0.3, alpha=0.6)
        ax[1].set_title("IQ trajectory")
    ax[1].set_aspect("equal")
    fig.tight_layout()
    return {"metrics": metrics, "fig": fig, "result": res, "waveform": wf}


def save_setup(path: str, preset: str, overrides: dict | None = None, *,
               seed: int = 1, noise: bool = True) -> None:
    """Persist a workbench setup as JSON: (preset name, overrides,
    seed, noise) — the complete recipe run_chain_report needs.  Only
    JSON-serializable overrides are accepted (all GUI knobs are)."""
    import json
    doc = {"polartx_setup": 1, "preset": preset,
           "overrides": overrides or {}, "seed": seed, "noise": noise}
    try:                                # fail fast on non-serializable
        json.dumps(doc)
    except TypeError as exc:
        raise ValueError(f"overrides not JSON-serializable: {exc}") from exc
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)


def load_setup(path: str) -> dict:
    """Load a setup saved by save_setup; validates the preset name."""
    import json
    with open(path) as f:
        doc = json.load(f)
    if doc.get("polartx_setup") != 1:
        raise ValueError(f"{path} is not a polartx setup file")
    if doc["preset"] not in PRESETS:
        raise ValueError(f"unknown preset {doc['preset']!r}")
    doc.setdefault("overrides", {})
    doc.setdefault("seed", 1)
    doc.setdefault("noise", True)
    return doc


def run_setup(doc: dict) -> dict:
    """Execute a loaded setup through run_chain_report."""
    return run_chain_report(doc["preset"], seed=int(doc["seed"]),
                            noise=bool(doc["noise"]), **doc["overrides"])


def run_mc_report(n_chips: int = 30, *, bw: float = 160e6,
                  skew_sigma_ns: float = 0.5, calibrated: bool = False,
                  limit_db: float = -35.0) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .montecarlo import run_mc, wifi_chip_builder

    mc = run_mc(wifi_chip_builder(bw=bw, n_symbols=3,
                                  skew_sigma_s=skew_sigma_ns * 1e-9,
                                  calibrated_skew=calibrated),
                n_chips, limit=limit_db)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(mc.values, bins=15, alpha=0.75)
    ax.axvline(limit_db, color="r", ls="--", label=f"limit {limit_db} dB")
    ax.set(xlabel="EVM [dB]", ylabel="chips",
           title=f"yield {100 * mc.yield_frac:.0f}% "
                 f"({'with' if calibrated else 'no'} skew cal)")
    ax.legend()
    fig.tight_layout()
    return {"summary": mc.summary(), "fig": fig}
