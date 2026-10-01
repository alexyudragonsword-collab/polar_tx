"""Example 09: real measured DPA data through the polar chain (OpenDPD).

Loads the OpenDPD DPA_160MHz / DPA_200MHz captures (real digital-PA
measurements, 640/800 MS/s), extracts the static polar characteristics
(binned AM-AM / AM-PM), and reports the number that matters: the
static-polar NMSE (~ -20 dB) versus OpenDPD's published GMP-510
(~ -39 dB) — the gap IS the device's memory, the part polar LUT DPD
cannot fix and the Cartesian ILA (ex08) exists for.  Then the WiFi-160
polar chain runs on the measured characteristics with the polar DPD.

Part 4 inverts that memory from the same capture as a RESIDUAL model
(measured.fit_residual_memory: GMP-510 trained on static-model output ->
measured output, chain scale) and reports the three NMSEs that settle
whether the 19 dB is memory — static LUT, LUT + memoryless polynomial
correction, LUT + full residual — then puts the measured DPA AND its
residual memory into the chain and shows what the polar LUT DPD and the
whole-chain ILA each recover.

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

fig, ax = plt.subplots(2, 2, figsize=(13, 9.2))
ax = ax.ravel()

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

print("\nthe remaining EVM/ACLR after polar DPD is the device's MEMORY "
      "- see ex08's\nwhole-chain Cartesian ILA for that part.")

# ----------------------- Part 4: the memory, inverted from the capture
from dataclasses import replace

from polartx.cal.memory_dpd import fit_chain_ila, run_with_ila
from polartx.measured import (fit_residual_memory, residual_training_pair,
                              spline_gmp_residual)
from polartx.vendor.padpd.pa import gmp_opendpd_510

print("\n=== residual memory models (validation segment, chain scale) ===")
hdr = f"{'device':12s} {'static LUT':>11s} {'+memoryless':>12s} {'+GMP-510':>9s} {'+SplineGMP':>11s}  cond GMP / spline"
print(hdr)
print("-" * len(hdr))
mem = {}
for name in ("DPA_160MHz", "DPA_200MHz"):
    _, ch = chars[name]
    model, rep = fit_residual_memory(ch)
    u, _ = residual_training_pair(ch)
    _, rep_s = fit_residual_memory(ch, lambda: spline_gmp_residual(u[:rep["n_train"]]))
    mem[name] = (model, rep)
    print(f"{name:12s} {rep['static_nmse_db']:11.2f} {rep['memoryless_nmse_db']:12.2f} "
          f"{rep['residual_nmse_db']:9.2f} {rep_s['residual_nmse_db']:11.2f}  "
          f"{rep['condition_number']:.1e} / {rep_s['condition_number']:.1e}")
print("(memoryless = a 5th-order static polynomial on top of the 64-bin LUT: the\n"
      " part of the static-vs-measured gap a finer static model could still buy)")

print("\n=== WiFi-160 chain: measured DPA_160MHz + its residual memory ===")
dpa, ch = chars["DPA_160MHz"]
model, rep = mem["DPA_160MHz"]
# drive the chain at the capture's full scale and sample rate: the residual
# model is only valid at the scale and rate it was fitted at
wf4 = wifi_waveform(160e6, 1024, n_symbols=4, seed=1)
wf4 = replace(wf4, x=wf4.x / np.abs(wf4.x).max() * ch["fs_scale"])
assert abs(wf4.fs - ch["fs"]) < 1.0
cfg = ChainConfig(env_floor=0.02, fs_scale_fixed=ch["fs_scale"])
rows = {}
for lab, dpd_, mem_ in (("static DPA, no DPD", None, None),
                        ("static DPA + polar DPD", PolarDPD.from_dpa(dpa), None),
                        ("DPA + memory, no DPD", None, model),
                        ("DPA + memory + polar DPD", PolarDPD.from_dpa(dpa), model)):
    r = PolarTX(cfg, IdealPhaseModulator(), dpa, dpd=dpd_, memory=mem_).run(wf4, noise=False)
    rows[lab] = (r.evm().db, aclr(r.y, r.fs, wf4.bw)["upper_dbc"])
txm = PolarTX(cfg, IdealPhaseModulator(), dpa, memory=model)
# the predistorter gets the residual's own structure (GMP-510); the
# default ILA model is smaller and recovers ~3 dB less here
ila = fit_chain_ila(txm, wf4, model_factory=gmp_opendpd_510)
r = run_with_ila(txm, wf4, ila, noise=False)
rows["DPA + memory + whole-chain ILA"] = (r.evm().db, aclr(r.y, r.fs, wf4.bw)["upper_dbc"])
for lab, (e, a_) in rows.items():
    print(f"{lab:32s} EVM {e:7.2f} dB  ACLR {a_:6.1f} dBc")
print("polar DPD buys ~18 dB on the static device and ~0 dB once the memory is in\n"
      "the chain: the memory is real and it is the ILA's job.")

a = ax[3]
u, v = residual_training_pair(ch)
n_tr = rep["n_train"]
err_static = np.abs(v - u)[n_tr:]
err_res = np.abs(v - model(u))[n_tr:]
amp = np.abs(u)[n_tr:]
order = np.argsort(amp)
nb = 40
edges = np.quantile(amp, np.linspace(0, 1, nb + 1))
idx = np.clip(np.digitize(amp, edges) - 1, 0, nb - 1)
centers = 0.5 * (edges[:-1] + edges[1:])
rms = lambda e: np.array([np.sqrt(np.mean(e[idx == b] ** 2)) for b in range(nb)])
a.semilogy(centers, rms(err_static), "o-", ms=3, label=f"static LUT only ({rep['static_nmse_db']:.1f} dB)")
a.semilogy(centers, rms(err_res), "s-", ms=3, label=f"+ residual GMP-510 ({rep['residual_nmse_db']:.1f} dB)")
a.set(xlabel="|static model output| (chain scale)", ylabel="rms error vs measured",
      title="DPA_160MHz: rms error vs measured, by amplitude")
a.grid(alpha=0.3, which="both")
a.legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex09_measured_dpa.png"), dpi=130)
print(f"plots -> {OUT}/ex09_measured_dpa.png")
