"""Example 08: the M4 suite — power-detector skew search, whole-chain
ILA-GMP memory DPD, Monte Carlo yield, and RTL export.

Part 1  ACP-search skew calibration (band-power observation only).
Part 2  Cartesian ILA-GMP DPD fitted against the ENTIRE polar chain
        (compressive DPA + AM-PM + linear/cubic memory): the
        fs_scale_fixed lesson — the chain must be static across runs.
Part 3  Monte Carlo: 40-chip populations, skew-cal off/on.
Part 4  polar-DPD dual-LUT RTL export + iverilog golden check.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx.cal.memory_dpd import fit_chain_ila, run_with_ila
from polartx.cal.polar_dpd import PolarDPD
from polartx.cal.skew import estimate_skew_by_acp
from polartx.chain import ChainConfig, PolarTX
from polartx.dpa import DPA, DPAConfig
from polartx.export.rtl import (emit_dpd_rtl, quantize_dpd_luts,
                                verify_with_iverilog)
from polartx.montecarlo import run_mc, wifi_chip_builder
from polartx.phasemod import IdealPhaseModulator
from polartx.presets import wifi_dtc
from polartx.vendor.padpd.metrics import aclr, psd
from polartx.waveforms.ofdm import wifi_waveform

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
fig, ax = plt.subplots(2, 2, figsize=(13, 9.2))

# ---------------------------------------- Part 1: ACP skew search
p = wifi_dtc(bw=160e6, env_skew_s=2.3e-9)
wf160 = p.make_waveform(n_symbols=4, seed=2)
est = estimate_skew_by_acp(p.tx, wf160, span_s=5e-9, n_grid=7)
print(f"ACP-search skew cal: injected 2.30 ns -> estimated "
      f"{est['skew_s'] * 1e9:.2f} ns (power detector only)")
a = ax[0, 0]
a.plot(est["trials_s"] * 1e9, est["acp_db"], "o-")
a.axvline(-2.3, ls=":", color="k", label="ideal correction")
a.set(xlabel="trial delay correction [ns]", ylabel="total ACP [dB]",
      title="skew cal with only a band-power detector")
a.legend()
a.grid(alpha=0.3)

# -------------------------------- Part 2: whole-chain ILA memory DPD
wf = wifi_waveform(80e6, 256, n_symbols=6, seed=1)
fir = np.array([1.0, 0.12 + 0.04j])


def memory(y):
    lin = np.convolve(y, fir, mode="full")[:y.size]
    return lin + 0.05 * lin * np.abs(lin) ** 2


tx = PolarTX(ChainConfig(env_floor=0.02,
                         fs_scale_fixed=1.3 * np.abs(wf.x).max()),
             IdealPhaseModulator(),
             DPA(DPAConfig(n_bits=11, amam=("rapp", 2.5, 1.2),
                           ampm_deg_poly=(0.0, 3.0, 4.0))),
             memory=memory)
r0 = tx.run(wf, noise=False)
dpd = fit_chain_ila(tx, wf)
r1 = run_with_ila(tx, wf, dpd, noise=False)
print(f"whole-chain ILA-GMP: EVM {r0.evm().db:.1f} -> {r1.evm().db:.1f} dB, "
      f"ACLR {aclr(r0.y, r0.fs, wf.bw)['upper_dbc']:.1f} -> "
      f"{aclr(r1.y, r1.fs, wf.bw)['upper_dbc']:.1f} dBc "
      f"({dpd.dpd_model.coeffs.size} coeffs)")
a = ax[0, 1]
for res, lab, c in ((r0, "memory + DPA, no DPD", "tab:red"),
                    (r1, "with whole-chain ILA-GMP", "tab:green")):
    f, pdb = psd(res.y, res.fs, nfft=8192)
    a.plot(f / 1e6, pdb, lw=0.7, color=c, label=lab)
a.set(xlim=(-320, 320), ylim=(-90, 5), xlabel="offset [MHz]",
      ylabel="dBr", title="Cartesian memory DPD around the polar chain")
a.legend(fontsize=8)

# ------------------------------------------- Part 3: Monte Carlo
mc_raw = run_mc(wifi_chip_builder(bw=160e6, skew_sigma_s=0.5e-9), 40,
                limit=-35.0)
mc_cal = run_mc(wifi_chip_builder(bw=160e6, skew_sigma_s=0.5e-9,
                                  calibrated_skew=True), 40, limit=-35.0)
print(f"Monte Carlo (40 chips, skew sigma 0.5 ns): yield "
      f"{100 * mc_raw.yield_frac:.0f}% raw -> "
      f"{100 * mc_cal.yield_frac:.0f}% with per-chip skew cal")
a = ax[1, 0]
bins = np.linspace(-42, -14, 24)
a.hist(mc_raw.values, bins=bins, alpha=0.6,
       label=f"raw (yield {100 * mc_raw.yield_frac:.0f}%)")
a.hist(mc_cal.values, bins=bins, alpha=0.6,
       label=f"skew-cal (yield {100 * mc_cal.yield_frac:.0f}%)")
a.axvline(-35, color="r", ls="--", label="limit -35 dB")
a.set(xlabel="EVM [dB]", ylabel="chips", title="mismatch Monte Carlo")
a.legend(fontsize=8)

# ------------------------------------------- Part 4: RTL export
luts = quantize_dpd_luts(PolarDPD.from_dpa(
    DPA(DPAConfig(n_bits=10, amam=("rapp", 2.5, 1.1),
                  ampm_deg_poly=(0.0, 2.0, 3.0)))))
rtl_dir = os.path.join(OUT, "rtl_dpd")
emit_dpd_rtl(luts, rtl_dir)
sim_out = verify_with_iverilog(rtl_dir)
if sim_out:
    verdict = next((l for l in sim_out.splitlines() if "PASS" in l
                    or "FAIL" in l), sim_out.strip().splitlines()[-1])
    print("RTL export: iverilog", verdict)
else:
    print("RTL export: emitted (iverilog not installed)")
a = ax[1, 1]
x = np.arange(luts["amp_code"].size) / (luts["amp_code"].size - 1)
a.step(x, luts["amp_float"], where="mid", label="AM-AM inverse LUT (12b)")
a2 = a.twinx()
a2.step(x, np.rad2deg(luts["ph_float"]), where="mid", color="tab:red",
        label="AM-PM corr LUT (14b)")
a2.set_ylabel("phase corr [deg]", color="tab:red")
a.set(xlabel="envelope command", ylabel="DPA code (norm)",
      title="exported dual-LUT DPD (256 x fixed-point)")
a.legend(loc="upper left", fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex08_m4_dpd_mc_rtl.png"), dpi=130)
print(f"plots -> {OUT}/ex08_m4_dpd_mc_rtl.png")
