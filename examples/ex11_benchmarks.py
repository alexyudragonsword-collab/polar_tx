"""Example 11: literature-class benchmarks (pllsim ex14 convention).

Three published digital polar transmitters, each modeled with
technology-plausible assumptions and landing in the published
performance CLASS — order-of-magnitude anchors, not chip reproductions:

- Staszewski et al., JSSC 2005: the first all-digital polar TX
  (GSM/EDGE, 90 nm).  3pi/8-8PSK @ 270.833 ksym/s through the ADPLL
  two-point path on the 26 MHz GSM crystal.  Published class: rms EVM
  ~2-3% (spec 9%).
- Madoglio et al., ISSCC 2014-class: 32 nm LTE-20 digital polar.
  Published class: EVM ~ -30 dB, E-UTRA ACLR limit -33 dBc met.
- 802.11n-era 20 MHz digital polar (Intel-class): published class EVM
  ~ -28 dB at 64-QAM (spec -25 dB).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx.metrics.aclr_ext import aclr_multi
from polartx.presets import (bench_edge_polar_staszewski05,
                             bench_lte20_polar_madoglio14,
                             bench_wifi11n_polar)
from polartx.waveforms.ofdm import demodulate_ofdm

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

print("=== literature-class digital polar TX benchmarks ===")
print(f"{'benchmark':>34s}{'simulated':>12s}{'published class':>18s}")

# ------------------------------------------------ EDGE (Staszewski'05)
p = bench_edge_polar_staszewski05()
wf = p.make_waveform(500, seed=1)
r = p.tx.run(wf, noise=True, seed=1)
d = r.evm()
print(f"{'Staszewski JSSC05 EDGE (DEVM)':>34s}{d['devm_pct']:10.2f}%"
      f"{'~2-3% (spec 9%)':>18s}")
z = d["symbols_rx"][20:-20]
ax[0].plot(z.real, z.imag, ".", ms=2, alpha=0.4, color="tab:blue")
ax[0].set_aspect("equal")
ax[0].set(title=f"EDGE 3π/8-8PSK — DEVM {d['devm_pct']:.2f}%",
          xlabel="I", ylabel="Q")

# ------------------------------------------ LTE-20 (Madoglio ISSCC'14)
p = bench_lte20_polar_madoglio14()
wf = p.make_waveform(n_symbols=10, seed=0)
r = p.tx.run(wf, noise=True, seed=1)
a = aclr_multi(r.y, r.fs, 20e6, offsets=(1,))
print(f"{'Madoglio ISSCC14 LTE-20 (EVM)':>34s}{r.evm().db:10.1f} dB"
      f"{'~-30 dB':>18s}")
print(f"{'  E-UTRA ACLR1':>34s}{a['aclr1_upper_dbc']:10.1f} dBc"
      f"{'limit -33 met':>18s}")
rx = demodulate_ofdm(r.y, wf.ofdm_ref)
g = np.vdot(wf.ofdm_ref.tx_symbols, rx) / np.vdot(wf.ofdm_ref.tx_symbols,
                                                  wf.ofdm_ref.tx_symbols)
pts = (rx / g).ravel()
ax[1].plot(pts.real, pts.imag, ".", ms=1, alpha=0.35, color="tab:red")
ax[1].set_aspect("equal")
ax[1].set(title=f"LTE-20 64-QAM — EVM {r.evm().db:.1f} dB",
          xlabel="I", ylabel="Q")

# ------------------------------------------------- WiFi 11n-era polar
p = bench_wifi11n_polar()
wf = p.make_waveform(n_symbols=8, seed=0)
r = p.tx.run(wf, noise=True, seed=1)
print(f"{'802.11n-era 20 MHz polar (EVM)':>34s}{r.evm().db:10.1f} dB"
      f"{'~-28 dB (spec -25)':>18s}")
rx = demodulate_ofdm(r.y, wf.ofdm_ref)
g = np.vdot(wf.ofdm_ref.tx_symbols, rx) / np.vdot(wf.ofdm_ref.tx_symbols,
                                                  wf.ofdm_ref.tx_symbols)
pts = (rx / g).ravel()
ax[2].plot(pts.real, pts.imag, ".", ms=1.5, alpha=0.35, color="tab:green")
ax[2].set_aspect("equal")
ax[2].set(title=f"11n 64-QAM — EVM {r.evm().db:.1f} dB",
          xlabel="I", ylabel="Q")

print("\nassumption labels are in each preset docstring; '-class' means "
      "order-of-magnitude\nanchor with technology-plausible parameters, "
      "not a chip reproduction.")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex11_benchmarks.png"), dpi=130)
print(f"plots -> {OUT}/ex11_benchmarks.png")
