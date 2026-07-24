"""Example 07: 5G NR wideband polar chains + the M3 calibration suite.

Part 1  NR FR1 100 MHz @ 3.5 GHz (30 kHz SCS, 256-QAM) and FR2 200 MHz
        @ 28 GHz (120 kHz SCS, 64-QAM) through the open-loop DTC polar
        chain: EVM / ACLR1 / stylized OBUE, constellations.
Part 2  Open-loop DTC gain/INL LUT calibration: CW training, spur table
        before/after (2 iterations), residual vs quantization floor.
Part 3  Background two-point sign-sign LMS (event ADPLL engine):
        convergence trace from a 5% direct-path gain error.
Part 4  DPA interleaving: first amplitude image combed to N x f_dpa.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx.cal.dtc_cal import apply_dtc_correction, fit_dtc_correction
from polartx.impairments import zoh_hold
from polartx.dpa import DPA, DPAConfig
from polartx.metrics.aclr_ext import aclr_multi
from polartx.phasemod import DTCPhaseModulator, DTCPMConfig
from polartx.presets import ble_1m_adpll, nr_dtc
from polartx.vendor.pllsim.calibration.lms import SignSignLMS
from polartx.waveforms.ofdm import demodulate_ofdm

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
fig, ax = plt.subplots(2, 2, figsize=(13, 9.6))

# ---------------------------------------------------------- Part 1: NR
print("=== 5G NR wideband polar chains ===")
nr_res = {}
for bw in (100e6, 200e6):
    p = nr_dtc(bw=bw)
    wf = p.make_waveform(n_symbols=6, seed=0)
    r = p.tx.run(wf, noise=True, seed=1)
    a = aclr_multi(r.y, r.fs, bw, offsets=(1,))
    ok, m, _ = r.check_mask()
    nr_res[bw] = (wf, r)
    band = "FR1 3.5 GHz" if bw <= 100e6 else "FR2 28 GHz"
    print(f"NR {bw / 1e6:3.0f} MHz ({band}): EVM {r.evm().db:6.1f} dB, "
          f"ACLR1 {a['aclr1_upper_dbc']:6.1f} dBc, "
          f"OBUE {'PASS' if ok else 'FAIL'}")

wf, r = nr_res[100e6]
rx = demodulate_ofdm(r.y, wf.ofdm_ref)
g = np.vdot(wf.ofdm_ref.tx_symbols, rx) / np.vdot(wf.ofdm_ref.tx_symbols,
                                                  wf.ofdm_ref.tx_symbols)
pts = (rx / g).ravel()
ax[0, 0].plot(pts.real, pts.imag, ".", ms=0.6, alpha=0.35,
              color="tab:blue")
ax[0, 0].set_aspect("equal")
ax[0, 0].set(title=f"NR FR1 100 MHz 256-QAM — EVM {r.evm().db:.1f} dB",
             xlabel="I", ylabel="Q")

# --------------------------------------------- Part 2: DTC LUT cal
FS, N = 640e6, 1 << 16
cw = 2 * np.pi * 5e6 * np.arange(N) / FS
pm = DTCPhaseModulator(DTCPMConfig(n_bits=12, gain_error=0.01,
                                   inl_sin=(2e-3, 3, 0.0),
                                   inl_poly=(0.0, 1e-3, -2e-3)))


def spec_dbc(pm_):
    out = pm_.modulate(cw, FS, noise=False).phase_out
    s = np.abs(np.fft.fft(np.exp(1j * out) * np.hanning(N)))
    return 20 * np.log10(np.maximum(s / s.max(), 1e-12))


db_before = spec_dbc(pm)
for _ in range(2):
    apply_dtc_correction(pm, fit_dtc_correction(pm, cw, FS))
db_after = spec_dbc(pm)
f_ax = np.fft.fftfreq(N, 1 / FS)
idx = np.argsort(f_ax)
a = ax[0, 1]
a.plot(f_ax[idx] / 1e6, db_before[idx], lw=0.6, label="before cal")
a.plot(f_ax[idx] / 1e6, db_after[idx], lw=0.6, label="after 2x fit/apply")
a.set(xlim=(-40, 40), ylim=(-110, 5), xlabel="offset [MHz]", ylabel="dBc",
      title="DTC gain/INL LUT cal: INL spurs -47 -> -91 dBc")
a.legend()
print("\nDTC LUT cal: worst INL spur "
      f"{db_before[3 * 512 - 4:3 * 512 + 5].max():.0f} -> "
      f"{db_after[3 * 512 - 4:3 * 512 + 5].max():.0f} dBc class")

# ------------------------------------ Part 3: background two-point LMS
p = ble_1m_adpll(mode="event", dp_gain=1.05)
p.tx.phasemod.dp_cal = SignSignLMS(init=1.05, mu=2e-5)
wfb = p.make_waveform(n_bits=2000, seed=5)
rb = p.tx.run(wfb, noise=True, seed=3)
tr = rb.info["phasemod"]["sim"].cal_traces["dp_gain"]
a = ax[1, 0]
a.plot(np.arange(tr.size) / 32e3, tr, lw=0.8)
a.axhline(1.0, ls=":", color="k")
a.set(xlabel="time [ms]", ylabel="direct-path gain",
      title=f"background sign-sign LMS: 1.05 -> "
            f"{tr[-tr.size // 10:].mean():.3f}, EVM {rb.evm()['evm_pct']:.1f}%")
a.grid(alpha=0.3)
print(f"two-point LMS: gain 1.05 -> {tr[-tr.size // 10:].mean():.4f}, "
      f"EVM {rb.evm()['evm_pct']:.2f}% (matched floor ~2.9%)")

# ------------------------------------------- Part 4: DPA interleaving
n = 1 << 14
hold = 8
dpa = DPA(DPAConfig(n_bits=12))
env = 0.7 + 0.2 * np.sin(2 * np.pi * 64 * np.arange(n) / n)
a = ax[1, 1]
for il, c in ((1, "tab:red"), (2, "tab:orange"), (4, "tab:green")):
    banks = [zoh_hold(dpa.encode(env), hold, k * (hold // il))
             for k in range(il)]
    y = np.mean([dpa(b, np.zeros(n)) for b in banks], axis=0)
    s = np.abs(np.fft.fft(y * np.hanning(n)))
    db = 20 * np.log10(np.maximum(s / s.max(), 1e-12))
    a.plot(np.arange(n // 2) / (n / 2), db[:n // 2], lw=0.7, color=c,
           label=f"interleave {il}")
a.set(xlabel="frequency [x fs/2]", ylabel="dBc", ylim=(-110, 5),
      title=f"DPA amplitude images, update fs/{hold}: interleaving combs "
            "the first image")
a.legend()

fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex07_nr_polar_and_cal.png"), dpi=130)
print(f"\nplots -> {OUT}/ex07_nr_polar_and_cal.png")
