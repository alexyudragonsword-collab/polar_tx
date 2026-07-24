"""Example 04: WiFi 6/7 wideband polar TX end to end.

CFR -> polar split (hole punching) -> 11-bit dithered DTC phase modulator
(locked WiFi-7-class LO) -> 10-bit DPA.  EVM/ACLR/mask across 80/160/320
MHz, CCDF before/after CFR, and the AM/PM skew sensitivity with the
estimate-and-correct loop closed.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx.cal import corrected_chain_config, estimate_env_skew
from polartx.presets import wifi_dtc
from polartx.vendor.padpd.cfr import cfr_clip_filter
from polartx.vendor.padpd.metrics import ccdf
from polartx.waveforms.ofdm import papr_db

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------- headline table
print("=== WiFi polar chain: EVM / ACLR / mask ===")
print(f"{'config':>22s}{'EVM [dB]':>10s}{'ACLR [dBc]':>12s}{'mask':>6s}")
for bw, qam in ((80e6, 1024), (160e6, 1024), (320e6, 4096)):
    p = wifi_dtc(bw=bw, qam=qam)
    wf = p.make_waveform(n_symbols=6, seed=0)
    res = p.tx.run(wf, noise=True, seed=1)
    a = res.aclr()
    ok, m, _ = res.check_mask()
    print(f"{bw / 1e6:7.0f} MHz {qam:5d}-QAM{res.evm().db:10.1f}"
          f"{max(a['lower_dbc'], a['upper_dbc']):12.1f}"
          f"{'PASS' if ok else 'FAIL':>6s}")

# ------------------------------------------------------ skew sensitivity
print("\n=== AM/PM skew sensitivity (160 MHz 1024-QAM) ===")
print(f"{'skew [ns]':>10s}{'EVM [dB]':>10s}{'after cal':>11s}")
p0 = wifi_dtc(bw=160e6)
wf160 = p0.make_waveform(n_symbols=6, seed=0)
skews = np.array([0.0, 0.5, 1.0, 2.0, 4.0]) * 1e-9
evm_skew, evm_cal = [], []
for t in skews:
    p = wifi_dtc(bw=160e6, env_skew_s=t)
    res = p.tx.run(wf160, noise=True, seed=1)
    evm_skew.append(res.evm().db)
    est = estimate_env_skew(res)
    p.tx.cfg = corrected_chain_config(p.tx.cfg, est["skew_s"])
    evm_cal.append(p.tx.run(wf160, noise=True, seed=1).evm().db)
    print(f"{t * 1e9:10.1f}{evm_skew[-1]:10.1f}{evm_cal[-1]:11.1f}")

# ------------------------------------------------------------------ plots
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

# constellation (160 MHz)
res = wifi_dtc(bw=160e6).tx.run(wf160, noise=True, seed=1)
from polartx.waveforms.ofdm import demodulate_ofdm
rx = demodulate_ofdm(res.y, wf160.ofdm_ref)
g = np.vdot(wf160.ofdm_ref.tx_symbols, rx) / np.vdot(
    wf160.ofdm_ref.tx_symbols, wf160.ofdm_ref.tx_symbols)
pts = (rx / g).ravel()
ax[0].plot(pts.real, pts.imag, ".", ms=1, alpha=0.4)
ax[0].set(title=f"160 MHz 1024-QAM, EVM {res.evm().db:.1f} dB",
          xlabel="I", ylabel="Q")
ax[0].set_aspect("equal")

# CCDF before/after CFR
wf = wf160
xc = cfr_clip_filter(wf.x, 8.5, wf.fs, wf.bw)
for sig, lab in ((wf.x, f"raw (PAPR {papr_db(wf.x):.1f} dB)"),
                 (xc, f"CFR 8.5 dB (PAPR {papr_db(xc):.1f} dB)")):
    lvl, prob = ccdf(sig, 12.0)
    ax[1].semilogy(lvl, prob, label=lab)
ax[1].set(xlabel="dB above average", ylabel="CCDF",
          title="crest-factor reduction", ylim=(1e-6, 1))
ax[1].grid(True, alpha=0.3)
ax[1].legend()

# skew chart
ax[2].plot(skews * 1e9, evm_skew, "o-", label="with skew")
ax[2].plot(skews * 1e9, evm_cal, "s-", label="after estimate+correct")
ax[2].set(xlabel="AM/PM skew [ns]", ylabel="EVM [dB]",
          title="skew sensitivity and calibration")
ax[2].grid(True, alpha=0.3)
ax[2].legend()

fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex04_wifi_polar_chain.png"), dpi=130)
print(f"\nplots -> {OUT}/ex04_wifi_polar_chain.png")
