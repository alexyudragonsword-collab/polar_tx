"""Example 02: why wideband polar is hard — bandwidth expansion.

Splitting a modulated signal into envelope and phase is nonlinear: both
components occupy several times the composite bandwidth, and every
envelope null makes the phase slew by ~pi in one sample.  Hole punching
(clamping the envelope at env_floor x rms) trades a bounded, exactly
computable EVM cost for a bounded phase-path bandwidth — the first
design decision of any wideband polar TX.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx.polar import bandwidth_expansion, polar_split
from polartx.vendor.padpd.metrics import psd
from polartx.waveforms.ofdm import wifi_waveform

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

wf = wifi_waveform(80e6, 1024, n_symbols=6, seed=1)
x, fs = wf.x, wf.fs

print("=== occupied bandwidth (99% power), WiFi 80 MHz ===")
print(f"{'env_floor':>10s}{'composite':>12s}{'envelope':>12s}{'phase':>12s}"
      f"{'clamp EVM':>12s}{'clamped':>9s}")
floors = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
rows = []
for fl in floors:
    b = bandwidth_expansion(x, fs, env_floor=fl)
    rows.append(b)
    print(f"{fl:10.2f}{b['bw_composite'] / 1e6:10.0f} M"
          f"{b['bw_env'] / 1e6:10.0f} M{b['bw_phase'] / 1e6:10.0f} M"
          f"{b['split_info']['clamp_evm_db']:11.1f} dB"
          f"{100 * b['split_info']['clamped_frac']:8.2f}%")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

# spectra of the polar components
env0, ph0, _ = polar_split(x, env_floor=0.02)
for sig, lab in ((x, "composite"), (env0 - env0.mean(), "envelope (AC)"),
                 (np.exp(1j * ph0), "phase path exp(jφ)")):
    f, p = psd(sig, fs, nfft=8192)
    ax[0].plot(f / 1e6, p, lw=0.8, label=lab)
ax[0].set(xlabel="frequency [MHz]", ylabel="dBr", ylim=(-90, 5),
          title="polar components vs composite (80 MHz OFDM)")
ax[0].legend()

# envelope trajectory near a null
env, _, _ = polar_split(x)
n0 = 100 + int(np.argmin(env[100:-100]))
t = np.arange(-60, 60)
ax[1].plot(t, env[n0 - 60: n0 + 60], label="envelope")
for fl in (0.02, 0.1):
    e2, _, _ = polar_split(x, env_floor=fl)
    ax[1].plot(t, e2[n0 - 60: n0 + 60], label=f"floor {fl:.2f}")
ax[1].set(xlabel="sample", ylabel="|x|", title="hole punching at a null")
ax[1].legend()

# the trade-off curve: what the clamp buys (DPA dynamic range) and costs
dr = [20 * np.log10(1.0 / max(fl, 1e-3)) for fl in floors[1:]]
evm_cost = [r["split_info"]["clamp_evm_db"] for r in rows[1:]]
ax[2].plot([100 * f for f in floors[1:]], evm_cost, "o-",
           label="clamp EVM cost")
ax2b = ax[2].twinx()
ax2b.plot([100 * f for f in floors[1:]], dr, "s--", color="tab:orange",
          label="env dynamic range needed")
ax[2].set(xlabel="env_floor [% of rms]", ylabel="clamp EVM cost [dB]",
          title="hole-punch design trade")
ax2b.set_ylabel("envelope dynamic range [dB]")
ax[2].grid(True, alpha=0.3)
h1, l1 = ax[2].get_legend_handles_labels()
h2, l2 = ax2b.get_legend_handles_labels()
ax[2].legend(h1 + h2, l1 + l2, loc="center right")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex02_polar_bandwidth_expansion.png"), dpi=130)
print("\nnote: magnitude-only hole punching bounds the ENVELOPE dynamic "
      "range and\nband, but the phase path keeps its pi-flips (the DTC "
      "wraps them mod 2pi);\nphase-trajectory smoothing is a follow-up "
      "milestone.")
print(f"plots -> {OUT}/ex02_polar_bandwidth_expansion.png")
