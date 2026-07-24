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
- Ben Bassat et al., ISSCC/JSSC 2020: a 27 dBm dual-band all-digital
  polar TX supporting 160 MHz for Wi-Fi 6 (28 nm, switched-capacitor
  digital PA + transformer combining).  Published class: 160 MHz
  1024-QAM (MCS11), raw EVM ~ -29 dB at 6 dB backoff, -40 dB with DPD.
  (The earlier 802.11n-era anchor, bench_wifi11n_polar, stays importable
  for historical comparison.)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx.metrics.aclr_ext import aclr_multi
from polartx.presets import (bench_edge_polar_staszewski05,
                             bench_lte20_polar_madoglio14,
                             bench_wifi6_polar_benbassat20,
                             bench_wifi7_polar_degani24)
from polartx.waveforms.ofdm import demodulate_ofdm

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
# generational sweep of digital polar TX: EDGE'05, LTE'14, WiFi6'20, WiFi7'24
fig, axg = plt.subplots(2, 2, figsize=(11, 9))
ax = axg.ravel()

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

# ---------------------- Intel Wi-Fi 6 160 MHz polar (Ben Bassat'20)
p_raw = bench_wifi6_polar_benbassat20(dpd=False)
wf = p_raw.make_waveform(n_symbols=8, seed=0)
r_raw = p_raw.tx.run(wf, noise=True, seed=1)
p = bench_wifi6_polar_benbassat20()          # DPD on
r = p.tx.run(wf, noise=True, seed=1)
print(f"{'BenBassat ISSCC20 WiFi6 raw EVM':>34s}{r_raw.evm().db:10.1f} dB"
      f"{'~-29 dB @6dB BO':>18s}")
print(f"{'  with polar DPD':>34s}{r.evm().db:10.1f} dB"
      f"{'-40 dB class':>18s}")
rx = demodulate_ofdm(r.y, wf.ofdm_ref)
g = np.vdot(wf.ofdm_ref.tx_symbols, rx) / np.vdot(wf.ofdm_ref.tx_symbols,
                                                  wf.ofdm_ref.tx_symbols)
pts = (rx / g).ravel()
ax[2].plot(pts.real, pts.imag, ".", ms=0.6, alpha=0.3, color="tab:green")
ax[2].set_aspect("equal")
ax[2].set(title=f"Wi-Fi 6 160 MHz 1024-QAM (Intel'20)\n"
                f"raw {r_raw.evm().db:.1f} → DPD {r.evm().db:.1f} dB",
          xlabel="I", ylabel="Q")

# ------------------- Intel Wi-Fi 7 320 MHz polar (Degani RFIC'24)
p = bench_wifi7_polar_degani24()
wf = p.make_waveform(n_symbols=6, seed=0)
r = p.tx.run(wf, noise=True, seed=1)
eff = r.avg_efficiency(p.tx.dpa)
print(f"{'Degani RFIC24 WiFi7 320MHz EVM':>34s}{r.evm().db:10.1f} dB"
      f"{'-38 dB class':>18s}")
print(f"{'  avg eta (SCPA 34.7% peak)':>34s}{100 * eff['eta_avg']:9.1f}%"
      f"{'@%.1fdB BO' % eff['backoff_db']:>18s}")
rx = demodulate_ofdm(r.y, wf.ofdm_ref)
g = np.vdot(wf.ofdm_ref.tx_symbols, rx) / np.vdot(wf.ofdm_ref.tx_symbols,
                                                  wf.ofdm_ref.tx_symbols)
pts = (rx / g).ravel()
ax[3].plot(pts.real, pts.imag, ".", ms=0.4, alpha=0.25, color="tab:purple")
ax[3].set_aspect("equal")
ax[3].set(title=f"Wi-Fi 7 320 MHz 4096-QAM (Intel'24)\n"
                f"EVM {r.evm().db:.1f} dB, avg η {100 * eff['eta_avg']:.0f}%",
          xlabel="I", ylabel="Q")

print("\nassumption labels are in each preset docstring; '-class' means "
      "order-of-magnitude\nanchor with technology-plausible parameters, "
      "not a chip reproduction.")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex11_benchmarks.png"), dpi=130)
print(f"plots -> {OUT}/ex11_benchmarks.png")
