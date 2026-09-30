"""Example 19: polar vs outphasing on the SAME WiFi 160 MHz 1024-QAM burst.

Same waveform, same CFR, same DTC phase modulator, same DPA model, same
seed -- the only difference is the decomposition: envelope-on-code +
one phase path (polar) against two constant-envelope phase paths summed
in a combiner (outphasing, isolated Wilkinson and Chireix).  One table:

    EVM | ACLR | mask OOB margin | average efficiency | share of theta > 80 deg

The first four are the metrics every chain here is scored with; the last
is outphasing's own sensitivity number -- how much of the burst sits
within 10 degrees of the branches cancelling.

Part 1  the table (also on stdout).
Part 2  spectra of the three outputs and of ONE outphasing branch: the
        branches are constant-envelope, so their spectrum is far wider
        than the channel -- the reason the phase path (and the combiner
        match) is everything in this topology.
Part 3  the theta distribution and the two combiner efficiency laws
        against it: isolated burns cos^2(theta), Chireix holds its two
        peaks at theta_c and 90 - theta_c.
Part 4  what each topology is sensitive to: polar vs AM/PM skew,
        outphasing vs branch phase mismatch, both against their budgets.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx import OutphasingCombiner, wifi_dtc, wifi_outphasing
from polartx.metrics.masks import default_mask_spec
from polartx.metrics.sem import check_sem
from polartx.outphasing import (OutphasingTX, outphasing_decompose,
                                phase_mismatch_evm_budget_db)
from polartx.vendor.padpd.cfr import cfr_clip_filter
from polartx.vendor.padpd.metrics import psd

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

BW, QAM, SEED = 160e6, 1024, 1
polar_p = wifi_dtc(bw=BW, qam=QAM)
wf = polar_p.make_waveform(n_symbols=8, seed=0)
dpa = polar_p.tx.dpa

chains = {
    "polar (DTC + DPA)": polar_p.tx,
    "outphasing, isolated": OutphasingTX(
        polar_p.tx, OutphasingCombiner(mode="isolated")),
    "outphasing, Chireix 60°": OutphasingTX(
        polar_p.tx, OutphasingCombiner(mode="chireix", chireix_theta_c_deg=60.0)),
}


def oob_margin_db(res):
    f, pdb = res.psd(nfft=8192)
    return check_sem(f, pdb, default_mask_spec(wf), channel_bw_hz=wf.bw
                     ).get("oob_margin_db", float("nan"))


# ------------------------------------------------------ Part 1: the table
print(f"=== WiFi {BW / 1e6:.0f} MHz {QAM}-QAM, 8 symbols, CFR "
      f"{polar_p.tx.cfg.cfr_papr_db} dB, seed {SEED} ===")
hdr = f"{'chain':26} {'EVM dB':>8} {'ACLR dBc':>9} {'OOB margin dB':>14} " \
      f"{'eta_avg %':>10} {'theta>80 %':>11}"
print(hdr)
print("-" * len(hdr))
results = {}
for name, tx in chains.items():
    r = tx.run(wf, noise=True, seed=SEED)
    results[name] = r
    eff = r.avg_efficiency(dpa)["eta_avg"] * 100
    th80 = (r.theta_stats()["frac_gt80"] * 100 if hasattr(r, "theta_stats")
            else float("nan"))
    print(f"{name:26} {r.evm().db:8.2f} {r.aclr()['upper_dbc']:9.1f} "
          f"{oob_margin_db(r):14.1f} {eff:10.1f} {th80:11.1f}")
print("(theta > 80 %: n/a for polar -- it has no outphasing angle)")

# ------------------------------------------ Part 2: spectra incl. a branch
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
a = ax[0, 0]
for name, r in results.items():
    f, p = psd(r.y, r.fs, nfft=8192)
    a.plot(f / 1e6, p, lw=0.7, label=name)
r_o = results["outphasing, Chireix 60°"]
f, p = psd(r_o.taps[0].y, r_o.fs, nfft=8192)
a.plot(f / 1e6, p, lw=0.7, alpha=0.6, label="one outphasing BRANCH (constant envelope)")
a.set(xlabel="offset [MHz]", ylabel="dBr", ylim=(-90, 5),
      title="outputs coincide; a single branch is far wider than the channel")
a.legend(fontsize=7)

# ------------------------------------ Part 3: theta distribution + laws
a = ax[0, 1]
theta_deg = np.rad2deg(r_o.theta)
a.hist(theta_deg, bins=90, range=(0, 90), density=True, alpha=0.6,
       label=f"theta of this burst (>80°: {r_o.theta_stats()['frac_gt80'] * 100:.0f} %)")
th = np.linspace(0, np.pi / 2, 400)
a2 = a.twinx()
for name, c in (("isolated", OutphasingCombiner(mode="isolated", combiner_loss_db=0.0)),
                ("Chireix 60°", OutphasingCombiner(mode="chireix", combiner_loss_db=0.0))):
    a2.plot(np.rad2deg(th), c.power_factor(th), lw=1.4, label=f"{name} power factor")
a.set(xlabel="outphasing angle theta [deg]", ylabel="density",
      title="where the burst spends its time vs where each combiner is efficient")
a2.set(ylabel="efficiency / eta_pa", ylim=(0, 1.05))
a.legend(loc="upper left", fontsize=8); a2.legend(loc="upper right", fontsize=8)

# ------------------------------- Part 4: each topology's own sensitivity
a = ax[1, 0]
skews_ns = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
e_polar = []
for s in skews_ns:
    p = wifi_dtc(bw=BW, qam=QAM, env_skew_s=s * 1e-9)
    e_polar.append(p.tx.run(wf, noise=True, seed=SEED).evm().db)
a.plot(skews_ns, e_polar, "o-", label="polar: AM/PM path skew")
a.set(xlabel="envelope-path skew [ns]", ylabel="EVM [dB]",
      title="polar's sharpest knob does nothing to outphasing (asserted in tests)")
e_o = wifi_outphasing(bw=BW, qam=QAM, env_skew_s=1e-9).tx.run(
    wf, noise=True, seed=SEED).evm().db
a.axhline(e_o, color="C1", ls="--", label=f"outphasing with 1 ns 'skew' = {e_o:.1f} dB (unchanged)")
a.legend(fontsize=8)

a = ax[1, 1]
deltas = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
x = cfr_clip_filter(wf.x, polar_p.tx.cfg.cfr_papr_db, wf.fs, wf.bw)
_, s2, _ = outphasing_decompose(x, np.abs(x).max())
e_meas, e_bud = [], []
for d in deltas:
    c = OutphasingCombiner(mode="chireix", phase_imbalance_deg=(0.0, d))
    e_meas.append(OutphasingTX(polar_p.tx, c).run(wf, noise=True, seed=SEED).evm().db)
    e_bud.append(phase_mismatch_evm_budget_db(x, s2, d, wf.fs, wf.bw) if d else np.nan)
a.plot(deltas, e_meas, "s-", label="measured EVM (with the chain's own floor)")
a.plot(deltas, e_bud, "x--", label="mismatch-only closed-form budget")
a.set(xlabel="branch phase mismatch [deg]", ylabel="EVM [dB]",
      title="outphasing's sharpest knob: branch phase mismatch")
a.legend(fontsize=8)
fig.tight_layout()
path = os.path.join(OUT, "ex19_outphasing_vs_polar.png")
fig.savefig(path, dpi=110)
print("saved", path)
