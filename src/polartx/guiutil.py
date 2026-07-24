"""Pure-computation layer behind the Streamlit GUI (testable headless).

Every function returns plain data + matplotlib figures; the GUI pages
only lay them out.  Mirrors the sibling repos' gui_core/services split.
"""
from __future__ import annotations

import numpy as np

PRESETS = ["BLE LE-1M", "BLE LE-2M", "BT EDR2 pi/4-DQPSK", "BT EDR3 8DPSK",
           "LTE 20 MHz", "WiFi 80 MHz", "WiFi 160 MHz", "WiFi 320 MHz",
           "NR FR1 100 MHz", "NR FR2 200 MHz"]


def build_preset(name: str, **overrides):
    """Preset registry -> TxPreset; overrides forwarded to the factory."""
    from . import presets as P
    if name == "BLE LE-1M":
        return P.ble_1m_adpll(**overrides)
    if name == "BLE LE-2M":
        return P.ble_2m_adpll(**overrides)
    if name == "BT EDR2 pi/4-DQPSK":
        return P.bt_edr_adpll("pi4dqpsk", **overrides)
    if name == "BT EDR3 8DPSK":
        return P.bt_edr_adpll("8dpsk", **overrides)
    if name == "LTE 20 MHz":
        return P.lte20_adpll(**overrides)
    if name.startswith("WiFi"):
        bw = float(name.split()[1]) * 1e6
        return P.wifi_dtc(bw=bw, **overrides)
    if name == "NR FR1 100 MHz":
        return P.nr_dtc(bw=100e6, **overrides)
    if name == "NR FR2 200 MHz":
        return P.nr_dtc(bw=200e6, **overrides)
    raise ValueError(f"unknown preset {name!r}")


def run_chain_report(name: str, *, seed: int = 1, noise: bool = True,
                     n_units: int | None = None, **overrides) -> dict:
    """Build, run, and package metrics + figures for one chain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .metrics import check_mask
    from .metrics.masks import default_mask

    p = build_preset(name, **overrides)
    wf = p.make_waveform(**({} if n_units is None else
                            ({"n_bits": n_units} if "BLE" in name else
                             {"n_syms": n_units} if "EDR" in name else
                             {"n_symbols": n_units})))
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
