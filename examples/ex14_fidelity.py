"""Example 14: waveform fidelity upgrades, quantified.

Part 1  SC-FDMA vs CP-OFDM (LTE uplink reality): 1.7 dB lower PAPR ->
        higher polar average efficiency for the same chain.
Part 2  Receiver-style EVM: preamble channel estimation + pilot CPE
        tracking versus the plain scalar-LS metric, under strong LO
        phase noise — how much the measurement convention matters.
Part 3  EDGE pulse fidelity: the linearized-GMSK C0 (numerically
        extracted Laurent pulse, self-checked) vs the old stylized
        SRRC — the real GMSK-like spectral skirt is ~35 dB wider at
        400 kHz, and the benchmark still lands in the published class
        with the spec's ideal-waveform EVM reference.
Part 4  BLE certification metrics: per-symbol delta-f2-max criterion
        and the modulation-index tolerance window.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx.metrics.ble_metrics import freq_deviation
from polartx.metrics.ofdm_rx import evm_rx
from polartx.presets import ble_1m_adpll, lte20_adpll, wifi_dtc
from polartx.vendor.padpd.metrics import evm_of_signal, psd
from polartx.vendor.pllsim.blocks.oscillator import OscConfig
from polartx.waveforms.edr import edge_waveform
from polartx.waveforms.ofdm import lte_waveform, wifi_waveform

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
fig, ax = plt.subplots(2, 2, figsize=(13, 9.2))

# ---------------------------------------------- Part 1: SC-FDMA
p = lte20_adpll()
res = {}
for sc in (False, True):
    wf = p.make_waveform(n_symbols=12, sc_fdma=sc)
    r = p.tx.run(wf, noise=True, seed=1)
    eff = r.avg_efficiency(p.tx.dpa)
    res[sc] = (wf, r, eff)
    print(f"LTE20 {'SC-FDMA(真实上行)' if sc else 'CP-OFDM(下行式)':>18s}: "
          f"PAPR {wf.meta['papr_db']:.1f} dB, EVM {r.evm().db:.1f} dB, "
          f"eta_avg {100 * eff['eta_avg']:.1f}%")
a = ax[0, 0]
for sc, c in ((False, "tab:red"), (True, "tab:green")):
    env = np.abs(res[sc][0].x)
    a.hist(20 * np.log10(np.maximum(env / np.sqrt(np.mean(env**2)), 1e-4)),
           bins=120, range=(-30, 12), histtype="step", color=c, lw=1.4,
           label=("SC-FDMA (UL)" if sc else "CP-OFDM") +
                 f", PAPR {res[sc][0].meta['papr_db']:.1f} dB")
a.set(xlabel="envelope [dB rel rms]", ylabel="samples", yscale="log",
      title="LTE20 envelope statistics")
a.legend(fontsize=8)

# ------------------------------- Part 2: receiver-style EVM under PN
noisy_lo = OscConfig(f0=5.9e9, gain=1.0, pn_dbchz=-102.0, pn_foffset=1e6,
                     pn_f1f3=400e3, pn_floor_dbchz=-145.0)
q = wifi_dtc(bw=80e6, qam=256, lo_pn=noisy_lo, lo_loop_bw=100e3)
wf = wifi_waveform(80e6, 256, n_symbols=8, seed=1, pilots=True, preamble=2)
r = q.tx.run(wf, noise=True, seed=1)
e_plain = evm_of_signal(r.y, wf.ofdm_ref).db
e_rx = evm_rx(r.y, wf).db
print(f"\n强相噪 LO 下的 EVM 口径: plain scalar {e_plain:.1f} dB, "
      f"receiver-style(前导+导频CPE) {e_rx:.1f} dB")
print("  -> 相噪快于符号率时导频 CPE 跟踪帮不上忙(估计自身带噪反而略差), "
      "与 stylized 口径的悲观度 <1 dB 的结论一致; 跟踪的价值在慢相位漂移"
      "场景(test_fidelity 里有确定性 CPE 的完全恢复证明)")
a = ax[0, 1]
a.bar(["plain scalar LS", "preamble+pilot CPE"], [e_plain, e_rx],
      color=["tab:red", "tab:green"], alpha=0.8)
a.set(ylabel="EVM [dB]",
      title="measurement convention under strong fast LO PN\n"
            "(pilot tracking cannot help; conventions within ~1 dB)")

# --------------------------------------------- Part 3: EDGE pulse
a = ax[1, 0]
for pulse, c in (("srrc", "tab:red"), ("lgmsk", "tab:green")):
    w = edge_waveform(300, pulse=pulse, seed=1)
    f, pdb = psd(w.x, w.fs, nfft=1 << 14)
    a.plot(f / 1e3, pdb, lw=0.7, color=c,
           label="stylized SRRC(0.3)" if pulse == "srrc"
                 else "linearized GMSK C0 (real EDGE)")
a.set(xlim=(-800, 800), ylim=(-100, 5), xlabel="offset [kHz]",
      ylabel="dBr", title="EDGE spectrum: the real GMSK-like skirt")
a.legend(fontsize=8)

# --------------------------------------------- Part 4: BLE cert metrics
p = ble_1m_adpll()
wfb = p.make_waveform(n_bits=600, pattern="10101010")
d = freq_deviation(p.tx.run(wfb, noise=True, seed=1).y, wfb)
print(f"\nBLE 认证口径: delta-f2-max P0.1 {d['df2max_p001_hz'] / 1e3:.0f} kHz "
      f"(>=185 需 99.9%: 实测 {100 * d['frac_above_185k']:.1f}%), "
      f"drift {d['drift_hz_per_us']:.1f} Hz/us")
a = ax[1, 1]
a.hist(d["df_sym_max_hz"] / 1e3, bins=40, alpha=0.8)
a.axvline(185, color="r", ls="--", label="185 kHz criterion")
a.set(xlabel="per-symbol max deviation [kHz]", ylabel="symbols",
      title=f"delta-f2-max distribution "
            f"({100 * d['frac_above_185k']:.1f}% above)")
a.legend()

fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex14_fidelity.png"), dpi=130)
print(f"\nplots -> {OUT}/ex14_fidelity.png")
