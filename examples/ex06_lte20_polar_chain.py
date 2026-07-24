"""Example 06: LTE 20 MHz narrowband polar TX end to end (M2).

ADPLL two-point at fref = fs = 122.88 MHz on the band-1 uplink carrier,
compressive 10-bit DPA with AM-PM, polar DPD (exact and measured-fit
LUTs), the offline two-point gain calibration, E-UTRA ACLR1/2 and the
stylized SEM, plus the analytic RX-band noise stack at the duplex offset
— the full cellular polar budget in one script.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx.cal.polar_dpd import PolarDPD
from polartx.cal.twopoint import estimate_dp_gain_error
from polartx.metrics import check_mask
from polartx.metrics.aclr_ext import aclr_multi
from polartx.metrics.masks import lte_sem
from polartx.metrics.rxband import adpll_rxband
from polartx.presets import lte20_adpll
from polartx.waveforms.ofdm import demodulate_ofdm

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

wf = lte20_adpll().make_waveform(n_symbols=20, seed=0)

# ------------------------------------------------ DPD on/off + fit
print("=== LTE20 polar chain (64/256-QAM grid), DPD study ===")
rows = []
for label, dpd_sel in (("no DPD", "off"), ("DPD exact LUT", "exact"),
                       ("DPD measured fit", "fit")):
    p = lte20_adpll(qam=64, dpd=(dpd_sel == "exact"))
    if dpd_sel == "fit":
        p0 = lte20_adpll(qam=64, dpd=False)
        meas = p0.tx.run(wf, noise=False)
        p.tx.dpd = PolarDPD.fit(meas)
    res = p.tx.run(wf, noise=True, seed=1)
    a = aclr_multi(res.y, res.fs, 20e6)
    ok, m, _ = res.check_mask()
    rows.append((label, res))
    print(f"{label:>18s}: EVM {res.evm().db:6.1f} dB, "
          f"ACLR1 {a['aclr1_upper_dbc']:6.1f}, ACLR2 {a['aclr2_upper_dbc']:6.1f} dBc, "
          f"SEM {'PASS' if ok else 'FAIL'}")

# ------------------------------------------------ two-point gain cal
p = lte20_adpll(qam=64, dp_gain=1.03)
r0 = p.tx.run(wf, noise=True, seed=2)
est = estimate_dp_gain_error(p.tx.phasemod, r0.phase_cmd, r0.phase_out, r0.fs)
p.tx.phasemod.dp_gain *= est["dp_gain_corr"]
r1 = p.tx.run(wf, noise=True, seed=2)
print(f"\ntwo-point cal: injected eps = 3%, estimated {100 * est['eps_hat']:.2f}% "
      f"-> EVM {r0.evm().db:.1f} -> {r1.evm().db:.1f} dB")

# ------------------------------------------------ RX-band noise stack
print("\nRX-band phase-noise budget (ADPLL path, analytic):")
for off, v in adpll_rxband(lte20_adpll().tx.phasemod,
                           offsets_hz=(30e6, 45e6, 60e6)).items():
    note = "  (beyond fref/2: indicative)" if v["beyond_fnyq"] else ""
    print(f"  {off / 1e6:5.0f} MHz: {v['ldbc_hz']:7.1f} dBc/Hz{note}")

# ------------------------------------------------------------- plots
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

a = ax[0]
sem = lte_sem(20e6)
for (label, res), c in zip(rows, ("tab:red", "tab:green", "tab:orange")):
    f, pdb = res.psd(nfft=8192)
    a.plot(f / 1e6, pdb, lw=0.7, color=c, label=label)
_, _, mask_db = check_mask(f, pdb, sem)
a.plot(f / 1e6, mask_db, "k--", lw=1, label="E-UTRA SEM (stylized)")
a.set(xlim=(-40, 40), ylim=(-90, 5), xlabel="offset [MHz]", ylabel="dBr",
      title="LTE20 spectrum: DPD closes ACLR")
a.legend(fontsize=8)

a = ax[1]
res = rows[1][1]
rx = demodulate_ofdm(res.y, wf.ofdm_ref)
g = np.vdot(wf.ofdm_ref.tx_symbols, rx) / np.vdot(wf.ofdm_ref.tx_symbols,
                                                  wf.ofdm_ref.tx_symbols)
pts = (rx / g).ravel()
a.plot(pts.real, pts.imag, ".", ms=1, alpha=0.3, color="tab:green")
a.set_aspect("equal")
a.set(xlabel="I", ylabel="Q",
      title=f"64-QAM w/ DPD: EVM {res.evm().db:.1f} dB")

a = ax[2]
dpa = lte20_adpll().tx.dpa
code = np.arange(dpa.cfg.n_codes) / (dpa.cfg.n_codes - 1)
a.plot(code, dpa.amp_table, label="DPA AM-AM (Rapp 2.5)")
dpd = PolarDPD.from_dpa(dpa)
pre, _ = dpd.predistort(code)
a.plot(code, dpa.amp_table[np.clip((pre * (dpa.cfg.n_codes - 1)).astype(int),
                                   0, dpa.cfg.n_codes - 1)],
       label="after DPD LUT")
a.plot(code, code, "k:", lw=0.8, label="ideal")
a2 = a.twinx()
a2.plot(code, np.rad2deg(dpa.phase_table), "--", color="tab:red",
        label="AM-PM [deg]")
a2.set_ylabel("AM-PM [deg]", color="tab:red")
a.set(xlabel="normalized code", ylabel="normalized amplitude",
      title="DPA characteristics and DPD inversion")
a.legend(loc="upper left", fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex06_lte20_polar_chain.png"), dpi=130)
print(f"\nplots -> {OUT}/ex06_lte20_polar_chain.png")
