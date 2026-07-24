"""Example 09: real measured DPA data through the polar chain (OpenDPD).

Loads the OpenDPD DPA_160MHz / DPA_200MHz captures (real digital-PA
measurements, 640/800 MS/s), extracts the static polar characteristics
(binned AM-AM / AM-PM), and reports the number that matters: the
static-polar NMSE (~ -20 dB) versus OpenDPD's published GMP-510
(~ -39 dB) — the gap IS the device's memory, the part polar LUT DPD
cannot fix and the Cartesian ILA (ex08) exists for.  Then the WiFi-160
polar chain runs on the measured characteristics with the polar DPD.

Needs an OpenDPD clone: git clone --depth 1
https://github.com/lab-emi/OpenDPD.git  (set $POLARTX_OPENDPD or place
it next to the repo).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx.cal.polar_dpd import PolarDPD
from polartx.chain import ChainConfig, PolarTX
from polartx.measured import find_opendpd_root, load_measured_dpa
from polartx.phasemod import IdealPhaseModulator
from polartx.vendor.padpd.metrics import aclr, psd
from polartx.waveforms.ofdm import wifi_waveform

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

if find_opendpd_root() is None:
    print("OpenDPD clone not found - set $POLARTX_OPENDPD; skipping.")
    sys.exit(0)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

# ------------------------------------------- extraction, both devices
print("=== OpenDPD measured DPA captures, static polar extraction ===")
chars = {}
for name in ("DPA_160MHz", "DPA_200MHz"):
    dpa, ch = load_measured_dpa(name)
    chars[name] = (dpa, ch)
    print(f"{name}: fs {ch['fs'] / 1e6:.0f} MS/s, gain {ch['gain']:.2f}, "
          f"AM-PM span {ch['ampm_deg'].max() - ch['ampm_deg'].min():.1f} deg, "
          f"static-polar NMSE {ch['static_nmse_db']:.1f} dB "
          f"(OpenDPD GMP-510 with memory: ~-39/-34 dB)")

a = ax[0]
for name, c in (("DPA_160MHz", "tab:blue"), ("DPA_200MHz", "tab:red")):
    _, ch = chars[name]
    x_a, y_a = ch["x_aligned"], ch["y_aligned"]
    sl = slice(0, 40000)
    a.plot(np.abs(x_a[sl]) / np.abs(x_a).max(),
           np.abs(y_a[sl]) / np.abs(y_a).max(), ".", ms=0.5, alpha=0.12,
           color=c)
    a.plot(ch["r_in"], ch["r_out"], "-", lw=2, color=c, label=name)
a.set(xlabel="normalized |x|", ylabel="normalized |y|",
      title="measured AM-AM (dots) and LUT fit")
a.legend()

a = ax[1]
for name, c in (("DPA_160MHz", "tab:blue"), ("DPA_200MHz", "tab:red")):
    _, ch = chars[name]
    a.plot(ch["r_in"], ch["ampm_deg"] - ch["ampm_deg"][-1], "-o", ms=3,
           color=c, label=name)
a.set(xlabel="normalized |x|", ylabel="AM-PM [deg]",
      title="measured AM-PM LUTs")
a.legend()
a.grid(alpha=0.3)

# ------------------------------- chain on the measured characteristics
print("\n=== WiFi-160 polar chain on the measured DPA_160MHz device ===")
dpa, ch = chars["DPA_160MHz"]
wf = wifi_waveform(160e6, 1024, n_symbols=6, seed=1)
a = ax[2]
for dpd, lab, c in ((None, "measured DPA, no DPD", "tab:red"),
                    (PolarDPD.from_dpa(dpa), "with polar LUT DPD",
                     "tab:green")):
    tx = PolarTX(ChainConfig(env_floor=0.02), IdealPhaseModulator(), dpa,
                 dpd=dpd)
    r = tx.run(wf, noise=False)
    f, pdb = psd(r.y, r.fs, nfft=8192)
    a.plot(f / 1e6, pdb, lw=0.7, color=c, label=lab)
    print(f"{lab:22s}: EVM {r.evm().db:6.1f} dB, "
          f"ACLR {aclr(r.y, r.fs, wf.bw)['upper_dbc']:6.1f} dBc")
a.set(xlim=(-640, 640), ylim=(-80, 5), xlabel="offset [MHz]",
      ylabel="dBr", title="chain spectrum, measured device")
a.legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex09_measured_dpa.png"), dpi=130)
print("\nthe remaining EVM/ACLR after polar DPD is the device's MEMORY "
      "- see ex08's\nwhole-chain Cartesian ILA for that part.")
print(f"plots -> {OUT}/ex09_measured_dpa.png")
