"""Example 10: DPA efficiency — the reason polar exists.

The SCPA's loss is switched-capacitor charging ∝ x(1−x) against output
power ∝ x², so drain efficiency rolls off far more gently at backoff
than a class-B linear PA (η ∝ x).  This chart set quantifies the
headline: average modulated efficiency per chain preset, the
efficiency-vs-backoff law, and the CFR-depth trade (deeper CFR = higher
average efficiency but more in-band clipping EVM).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx.chain import ChainConfig, PolarTX
from polartx.dpa import DPA, DPAConfig
from polartx.dpa.characteristics import efficiency_curve
from polartx.phasemod import IdealPhaseModulator
from polartx.presets import (ble_1m_adpll, bt_edr_adpll, lte20_adpll,
                             wifi_dtc)
from polartx.waveforms.ofdm import wifi_waveform

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

# ------------------------------------------- efficiency vs backoff law
x = np.linspace(0.01, 1.0, 300)
bo = -20 * np.log10(x)
a = ax[0]
for spec, lab in ((("scpa", 0.67, 0.85), "SCPA class-D (γ=0.67)"),
                  (("scpa", 0.3, 0.85), "SCPA, lower loss (γ=0.3)"),
                  (("classb", 0.85), "class-B linear PA")):
    a.plot(bo, 100 * efficiency_curve(spec, x), label=lab)
a.set(xlabel="output backoff [dB]", ylabel="drain efficiency [%]",
      xlim=(0, 20), title="efficiency vs backoff")
a.grid(alpha=0.3)
a.legend()

# --------------------------------- average efficiency per chain preset
print("=== modulated average drain efficiency (eta_peak = 85%) ===")
print(f"{'chain':>26s}{'PAPR/backoff':>14s}{'eta_avg':>9s}")
rows = []
cases = [
    ("BLE LE-1M (const env)", ble_1m_adpll(), dict(n_bits=400)),
    ("BT EDR3 8DPSK", bt_edr_adpll("8dpsk"), dict(n_syms=400)),
    ("LTE20 64QAM + DPD", lte20_adpll(), dict(n_symbols=8)),
    ("WiFi160 1024QAM CFR8.5", wifi_dtc(bw=160e6), dict(n_symbols=4)),
]
for label, p, wf_kw in cases:
    wf = p.make_waveform(**wf_kw)
    res = p.tx.run(wf, noise=False)
    e = res.avg_efficiency(p.tx.dpa)
    rows.append((label, e))
    print(f"{label:>26s}{e['backoff_db']:11.1f} dB{100 * e['eta_avg']:8.1f}%")
a = ax[1]
labels = [r[0].split(" (")[0] for r in rows]
a.bar(labels, [100 * r[1]["eta_avg"] for r in rows], color="tab:green",
      alpha=0.8)
a.axhline(85, ls=":", color="k", label="peak η")
a.set(ylabel="average drain efficiency [%]",
      title="modulated averages per preset")
a.tick_params(axis="x", rotation=20)
a.legend()

# --------------------------------------------- CFR depth trade (WiFi)
wf = wifi_waveform(160e6, 1024, n_symbols=4, seed=1)
paprs = [None, 10.0, 9.0, 8.5, 8.0, 7.0, 6.0]
eff_v, evm_v = [], []
for papr in paprs:
    tx = PolarTX(ChainConfig(env_floor=0.02, cfr_papr_db=papr),
                 IdealPhaseModulator(), DPA(DPAConfig(n_bits=10)))
    res = tx.run(wf, noise=False)
    eff_v.append(100 * res.avg_efficiency(tx.dpa)["eta_avg"])
    evm_v.append(res.evm().db)
a = ax[2]
a.plot([12.0 if p is None else p for p in paprs], eff_v, "o-",
       color="tab:green")
a.set(xlabel="CFR PAPR target [dB] (12 = no CFR)",
      ylabel="average efficiency [%]", title="CFR depth: efficiency vs EVM")
a2 = a.twinx()
a2.plot([12.0 if p is None else p for p in paprs], evm_v, "s--",
        color="tab:red")
a2.axhline(-35, ls=":", color="r")
a2.set_ylabel("EVM [dB]", color="tab:red")
a.grid(alpha=0.3)
print("\nCFR trade (WiFi160):",
      ", ".join(f"{'no' if p is None else p}dB→η{e:.0f}%/{v:.0f}dB"
                for p, e, v in zip(paprs, eff_v, evm_v)))

fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex10_dpa_efficiency.png"), dpi=130)
print(f"plots -> {OUT}/ex10_dpa_efficiency.png")
