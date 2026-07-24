"""Example 15: mixed-domain 2-tap FIR + digital-Doherty polar DTX for
WiFi MLO coexistence (Borokhovich, Socher & Degani, RFIC 2026).

Two DPA arrays (taps), each a full DTC-phase-modulated polar chain, are
fed the same codes with a programmable delay and coherently combined:
a 2-tap FIR H = 1 + exp(-j*omega*D) with |H|^2 = 4cos^2(pi f D/f_s).
The signal at 0 offset gets +6 dB; deep NOTCHES land at f_s/(2D) — put
one on the co-located MLO receiver's channel to null the transmitter's
out-of-channel (OOC) noise.

Part 1  the FIR notch and its configurability (delay -> notch offset).
Part 2  OOC suppression: on the deterministic quantization/DPD-residual
        content the notch is deep (~25 dB); with the random LO/jitter
        floor included it is shallower — the FIR only notches CORRELATED
        content, so the achievable depth is set by the random floor at
        the offset (the engineering lesson).
Part 3  constant BW*delta-f: wider signals need a proportionally closer
        notch (slide 24).
Part 4  the digital-Doherty efficiency bump at 6 dB backoff.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx.dpa.characteristics import efficiency_curve
from polartx.fir import ooc_noise_suppression_db
from polartx.presets import bench_wifi7_mlo_fir_borokhovich26
from polartx.vendor.padpd.metrics import psd

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
fig, ax = plt.subplots(2, 2, figsize=(13, 9.2))

# ---------------------------------------------- Part 1: notch spectra
print("=== FIR notch, WiFi 40 MHz OFDM at 6.025 GHz (offset from f0) ===")
wf = bench_wifi7_mlo_fir_borokhovich26(bw=40e6).make_waveform(
    n_symbols=3, seed=0)
a = ax[0, 0]
for off, c in ((500e6, "tab:red"), (800e6, "tab:green")):
    p = bench_wifi7_mlo_fir_borokhovich26(bw=40e6, notch_offset_hz=off)
    rf = p.fir_tx.run(wf, noise=False)
    f, pdb = psd(rf.y, rf.fs, nfft=1 << 14)
    a.plot(f / 1e6, pdb, lw=0.6, color=c, label=f"notch @ {off/1e6:.0f} MHz")
f, pdb = psd(p.single_tx.run(wf, noise=False).y, wf.fs, nfft=1 << 14)
a.plot(f / 1e6, pdb, lw=0.6, color="0.6", label="single tap (no FIR)")
a.set(xlim=(-1000, 1000), ylim=(-120, 5), xlabel="offset from f0 [MHz]",
      ylabel="dBc", title="2-tap FIR notch, configurable offset")
a.legend(fontsize=8)

# ------------------------------------ Part 2: deterministic vs realistic
print(f"{'notch offset':>14s}{'deterministic':>15s}{'realistic':>12s}{'EVM':>9s}")
a = ax[0, 1]
offs = np.array([300, 400, 500, 650, 800]) * 1e6
r1n = bench_wifi7_mlo_fir_borokhovich26(bw=40e6).single_tx.run(
    wf, noise=True, seed=1)
det, real = [], []
for off in offs:
    p = bench_wifi7_mlo_fir_borokhovich26(bw=40e6, notch_offset_hz=off)
    band = (off - 40e6, off + 40e6)
    det.append(ooc_noise_suppression_db(p.fir_tx.run(wf, noise=False),
                                        p.single_tx.run(wf, noise=False),
                                        band))
    rf = p.fir_tx.run(wf, noise=True, seed=1)
    real.append(ooc_noise_suppression_db(rf, r1n, band))
    print(f"{off/1e6:11.0f} MHz{det[-1]:12.1f} dB{real[-1]:9.1f} dB"
          f"{rf.evm().db:8.1f} dB")
a.plot(offs / 1e6, det, "o-", label="deterministic content")
a.plot(offs / 1e6, real, "s-", label="realistic (+ random floor)")
a.set(xlabel="notch offset [MHz]", ylabel="OOC suppression [dB]",
      title="notch depth: correlated content only\n(random floor sets the limit)")
a.legend(fontsize=8)
a.grid(alpha=0.3)

# --------------------------------- Part 3: constant BW*delta-f (slide 24)
a = ax[1, 0]
print("\nconstant BW*delta-f (equal notch-BW per use case):")
for bw, off, c in ((40e6, 800e6, "tab:red"), (80e6, 400e6, "tab:blue"),
                   (160e6, 200e6, "tab:green")):
    p = bench_wifi7_mlo_fir_borokhovich26(bw=bw, notch_offset_hz=off, osr=40)
    w = p.make_waveform(n_symbols=3, seed=0)
    f, pdb = psd(p.fir_tx.run(w, noise=False).y, w.fs, nfft=1 << 14)
    a.plot(f / 1e6, pdb, lw=0.6, color=c,
           label=f"BW {bw/1e6:.0f} MHz, notch {off/1e6:.0f} MHz")
    print(f"  BW {bw/1e6:.0f} MHz -> notch {off/1e6:.0f} MHz "
          f"(BW*offset = {bw*off/1e18:.2f}e18)")
a.set(xlim=(-1000, 1000), ylim=(-120, 5), xlabel="offset from f0 [MHz]",
      ylabel="dBc", title="wider signal -> closer notch (constant BW·Δf)")
a.legend(fontsize=8)

# ---------------------------------------- Part 4: Doherty efficiency
a = ax[1, 1]
x = np.linspace(0.02, 1.0, 300)
bo = -20 * np.log10(x)
a.plot(bo, 100 * efficiency_curve(("doherty", 0.55, 0.35, 6.0), x),
       label="digital Doherty (6 dB BO)")
a.plot(bo, 100 * efficiency_curve(("scpa", 0.55, 0.35), x), "--",
       label="plain SCPA")
a.axvline(6, ls=":", color="k")
a.set(xlim=(0, 20), xlabel="output backoff [dB]",
      ylabel="drain efficiency [%]",
      title="digital-Doherty efficiency bump at 6 dB backoff")
a.legend(fontsize=8)
a.grid(alpha=0.3)
i6 = np.argmin(np.abs(x - 0.5))
print(f"\nDoherty efficiency @6 dB backoff: "
      f"{100 * efficiency_curve(('doherty', 0.55, 0.35, 6.0), x)[i6]:.0f}% "
      f"vs plain SCPA {100 * efficiency_curve(('scpa', 0.55, 0.35), x)[i6]:.0f}%")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex15_fir_notch_mlo.png"), dpi=130)
print(f"\nplots -> {OUT}/ex15_fir_notch_mlo.png")
