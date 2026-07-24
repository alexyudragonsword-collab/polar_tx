"""Example 01: BLE GFSK through the ADPLL two-point polar TX.

The narrowband flavor end to end: LE-1M/LE-2M GFSK -> ADPLL two-point
phase modulation (loop BW ~100 kHz, data rate 10-20x above it) -> DPA at
constant envelope.  Shows

- the transmitted frequency trajectory and its eye,
- EVM vs direct-path gain mismatch (the two-point calibration spec),
  fast linearized "response" mode cross-checked against the
  cycle-accurate "event" engine,
- output PSD against the (stylized) BLE in-band emission mask,
- the ADPLL phase-noise budget behind the noise-limited EVM floor.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx.metrics import check_mask
from polartx.metrics.ble_metrics import freq_deviation
from polartx.metrics.masks import ble_mask
from polartx.presets import ble_1m_adpll, ble_2m_adpll

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

TWOPI = 2.0 * np.pi
ERRS = (0.0, 0.01, 0.02, 0.05, 0.10)

# ------------------------------------------------- EVM vs dp-gain mismatch
print("=== BLE LE-1M, ADPLL two-point, EVM vs direct-path gain error ===")
print(f"{'eps':>6s}{'response EVM':>14s}{'event EVM':>12s}")
wf = ble_1m_adpll().make_waveform(n_bits=600, seed=5)
evm_r, evm_e = [], []
for eps in ERRS:
    er = ble_1m_adpll(dp_gain=1 + eps).tx.run(wf, seed=3).evm()
    ee = ble_1m_adpll(dp_gain=1 + eps, mode="event").tx.run(wf, seed=3).evm()
    evm_r.append(er)
    evm_e.append(ee)
    print(f"{100 * eps:5.0f}%{er['evm_pct']:12.2f}%{ee['evm_pct']:11.2f}%")
print("matched EVM is phase-noise-limited; the loop BW plays no role at "
      "1 Mb/s\n")

# ------------------------------------------------------ deviation metrics
for name, preset in (("LE-1M", ble_1m_adpll()), ("LE-2M", ble_2m_adpll())):
    row = {}
    for pat in ("11110000", "10101010"):
        w = preset.make_waveform(n_bits=400, pattern=pat)
        d = freq_deviation(preset.tx.run(w, seed=1).y, w)
        row.update({k: v for k, v in d.items() if k.startswith("df")})
    print(f"{name}: df1_avg = {row['df1_avg_hz'] / 1e3:.1f} kHz, "
          f"df2_avg = {row['df2_avg_hz'] / 1e3:.1f} kHz, "
          f"df2_min = {row['df2_min_hz'] / 1e3:.1f} kHz")

# ------------------------------------------------------------------ plots
res = ble_1m_adpll().tx.run(wf, seed=3)
fs = res.fs
fig, ax = plt.subplots(2, 2, figsize=(12, 8.4))

# transmitted frequency trajectory + eye
f_inst = np.gradient(np.unwrap(np.angle(res.y))) * fs / TWOPI
sps = int(fs / 1e6)
a = ax[0, 0]
a.plot(np.arange(4000) / sps, f_inst[1000:5000] / 1e3, lw=0.8)
a.set(xlabel="symbol", ylabel="freq deviation [kHz]",
      title="LE-1M transmitted frequency trajectory")
a = ax[0, 1]
seg = f_inst[1000:1000 + 200 * 2 * sps].reshape(-1, 2 * sps)
for s in seg[:120]:
    a.plot(np.arange(2 * sps) / sps, s / 1e3, color="tab:blue", alpha=0.12)
a.set(xlabel="symbol", ylabel="freq deviation [kHz]", title="frequency eye")

# EVM vs mismatch design chart
a = ax[1, 0]
x = [100 * e for e in ERRS]
a.plot(x, [e["evm_pct"] for e in evm_r], "o-", label="response mode")
a.plot(x, [e["evm_pct"] for e in evm_e], "s--", label="event engine")
a.set(xlabel="direct-path gain error [%]", ylabel="EVM [%]",
      title="two-point mismatch design chart")
a.grid(True, alpha=0.3)
a.legend()

# PSD vs BLE mask
a = ax[1, 1]
f, p = res.psd(nfft=8192)
ok, margin, mask_db = check_mask(f, p, ble_mask(1e6))
a.plot(f / 1e6, p, lw=0.8, label="TX PSD")
a.plot(f / 1e6, mask_db, "r--", lw=1.0, label="BLE mask (stylized)")
a.set(xlabel="offset [MHz]", ylabel="dBr", xlim=(-5, 5), ylim=(-80, 5),
      title=f"emission mask: {'PASS' if ok else 'FAIL'}")
a.legend()

fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex01_ble_gfsk_adpll.png"), dpi=130)
print(f"\nmask {'PASS' if ok else 'FAIL'} (worst margin {margin:.1f} dB); "
      f"plots -> {OUT}/ex01_ble_gfsk_adpll.png")
