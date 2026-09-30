"""Example 20: polar vs outphasing vs Cartesian (RF-DAC) on the SAME burst.

Stage 2 of the three-topology comparison.  ex19 put outphasing beside the
polar chain; this adds the digital I/Q transmitter — two current-cell arrays
driven by I and Q codes, no envelope path, no phase path — built from the
same ``ChainConfig`` (only ``cfr_papr_db`` is read; everything else names
a path it does not have and is warned about) and the same unit-cell
mismatch model as the polar DPA.  One table:

    EVM | ACLR | mask OOB margin | average efficiency | 1-dB tolerance

The last column is each topology's tolerance to ITS OWN sharpest knob:
the largest value of that knob that still keeps the EVM within 1 dB of the
chain's unimpaired number (bisection on a 4-symbol burst) — AM/PM path
skew for polar, branch phase mismatch for outphasing, I/Q phase error for
the RF-DAC.  Three different units, on purpose: they are the numbers a
calibration has to hit, and they are not comparable across topologies.

Part 1  the table (also on stdout), real impairments, noise on.
Part 2  spectra of the three outputs.
Part 3  efficiency vs CW backoff for every law in the library: the polar
        SCPA, the RF-DAC on an axis / on the diagonal / phase-averaged,
        and the two outphasing combiners mapped through A/A_max = cos θ.
        The RF-DAC pays for |I| + |Q| and delivers I² + Q², so it sits
        below the SCPA at every backoff — the topology's known price.
Part 4  the RF-DAC's own sharpest knob: I/Q imbalance.  EVM against the
        phase error at a fixed gain error, next to the closed-form image
        rejection — the image lands in band, so EVM tracks −IRR until the
        CFR floor takes over.
"""
import os
import warnings
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx import (CartesianTX, ChainConfig, OutphasingCombiner, OutphasingTX,
                     RFDAC, RFDACConfig, iq_image_rejection_db, wifi_dtc,
                     wifi_rfdac)
from polartx.dpa.dpa import efficiency_curve
from polartx.metrics.masks import default_mask_spec
from polartx.metrics.sem import check_sem

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

BW, QAM, SEED = 160e6, 1024, 1
polar_p = wifi_dtc(bw=BW, qam=QAM)
rfdac_p = wifi_rfdac(bw=BW, qam=QAM)          # 12 bit, 0.2 % mismatch, 50 fs
wf = polar_p.make_waveform(n_symbols=8, seed=0)
dpa = polar_p.tx.dpa

chains = {
    "polar (DTC + DPA)": polar_p.tx,
    "outphasing, Chireix 60°": OutphasingTX(
        polar_p.tx, OutphasingCombiner(mode="chireix", chireix_theta_c_deg=60.0)),
    "Cartesian (RF-DAC 12 bit)": rfdac_p.tx,
}


def oob_margin_db(res):
    f, pdb = res.psd(nfft=8192)
    return check_sem(f, pdb, default_mask_spec(wf), channel_bw_hz=wf.bw
                     ).get("oob_margin_db", float("nan"))


def eta_avg(res):
    # polar / outphasing results take the DPA; the Cartesian one has its DAC
    r = res.avg_efficiency() if hasattr(res, "rfdac") else res.avg_efficiency(dpa)
    return r["eta_avg"]


# ----------------------------------- each topology's sharpest knob
CFR = polar_p.tx.cfg.cfr_papr_db
wf4 = polar_p.make_waveform(n_symbols=4, seed=0)     # the tolerance search burst


def polar_with_skew(s_ns):
    return wifi_dtc(bw=BW, qam=QAM, env_skew_s=s_ns * 1e-9).tx


def outphasing_with_mismatch(d_deg):
    return OutphasingTX(polar_p.tx, OutphasingCombiner(
        mode="chireix", chireix_theta_c_deg=60.0, phase_imbalance_deg=(0.0, d_deg)))


def rfdac_with_iq_phase(d_deg):
    return CartesianTX(ChainConfig(cfr_papr_db=CFR),
                       RFDAC(replace(rfdac_p.tx.rfdac.cfg, iq_phase_deg=d_deg)))


def tolerance_1db(make_tx, hi, n_iter=8):
    """Largest knob value whose EVM stays within 1 dB of the knob-at-zero
    EVM (same burst, same seed): bisection between 0 and ``hi``."""
    evm0 = make_tx(0.0).run(wf4, noise=True, seed=SEED).evm().db
    lo = 0.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if make_tx(mid).run(wf4, noise=True, seed=SEED).evm().db <= evm0 + 1.0:
            lo = mid
        else:
            hi = mid
    return lo


knobs = {
    "polar (DTC + DPA)": ("AM/PM skew", "ns", polar_with_skew, 2.0),
    "outphasing, Chireix 60°": ("branch phase", "deg", outphasing_with_mismatch, 10.0),
    "Cartesian (RF-DAC 12 bit)": ("I/Q phase", "deg", rfdac_with_iq_phase, 10.0),
}

# ------------------------------------------------------ Part 1: the table
print(f"=== WiFi {BW / 1e6:.0f} MHz {QAM}-QAM, 8 symbols, CFR {CFR} dB, "
      f"seed {SEED}, noise on ===")
hdr = (f"{'chain':28} {'EVM dB':>8} {'ACLR dBc':>9} {'OOB margin dB':>14} "
       f"{'eta_avg %':>10}  1-dB tolerance of its sharpest knob")
print(hdr)
print("-" * len(hdr))
results, tol = {}, {}
for name, tx in chains.items():
    r = tx.run(wf, noise=True, seed=SEED)
    results[name] = r
    label, unit, make, hi = knobs[name]
    tol[name] = tolerance_1db(make, hi)
    print(f"{name:28} {r.evm().db:8.2f} {r.aclr()['upper_dbc']:9.1f} "
          f"{oob_margin_db(r):14.1f} {eta_avg(r) * 100:10.1f}  "
          f"{label} {tol[name]:.2f} {unit}")
print("(RF-DAC impairments in this row: 0.2 % cell mismatch on both arrays, "
      "50 fs LO jitter, the plan's LO phase noise; no I/Q imbalance, no LO "
      "leakage -- opt-in knobs)")

# ------------------------------------------------ Part 2: spectra
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
a = ax[0, 0]
for name, r in results.items():
    f, p = r.psd(nfft=8192)
    a.plot(f / 1e6, p, lw=0.7, label=name)
a.set(xlabel="offset [MHz]", ylabel="dBr", ylim=(-90, 5),
      title="three topologies, one waveform")
a.legend(fontsize=8)

# ------------------------------ Part 3: efficiency vs CW backoff, all laws
a = ax[0, 1]
bo = np.linspace(0, 12, 200)
x = 10 ** (-bo / 20)                                  # amplitude / full scale
eta_pk = dpa.cfg.eff[2]
rf = RFDAC(RFDACConfig(eff=("iq_cells", eta_pk)))
a.plot(bo, 100 * efficiency_curve(dpa.cfg.eff, x), lw=1.8, label="polar SCPA law")
a.plot(bo, 100 * rf.efficiency(x, 0.0), lw=1.4, label="RF-DAC, on an axis")
a.plot(bo, 100 * rf.efficiency(x / np.sqrt(2), x / np.sqrt(2)), lw=1.4,
       label="RF-DAC, on the diagonal")
ph = np.linspace(0, 2 * np.pi, 720, endpoint=False)
eta_ph = np.array([np.mean(rf.efficiency(xx * np.cos(ph), xx * np.sin(ph))) for xx in x])
a.plot(bo, 100 * eta_ph, "k--", lw=1.2, label="RF-DAC, phase-averaged")
theta = np.arccos(np.clip(x, 0, 1))                  # A/A_max = cos theta
for mode in ("isolated", "chireix"):
    c = OutphasingCombiner(mode=mode, combiner_loss_db=0.0)
    a.plot(bo, 100 * eta_pk * c.power_factor(theta), lw=1.0, ls=":",
           label=f"outphasing {mode} (eta_pa = eta_peak)")
a.axvline(6.0, color="grey", lw=0.8)
a.set(xlabel="CW backoff from full scale [dB]", ylabel="drain efficiency [%]",
      ylim=(0, 100), title="efficiency laws vs backoff (same eta_peak)")
a.legend(fontsize=7)

# ------------------------------------- Part 4: I/Q imbalance sweep
a = ax[1, 0]
phis = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
G_DB = 0.1
e_meas, irr = [], []
for p_deg in phis:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # sweep builds many chains
        tx = CartesianTX(ChainConfig(cfr_papr_db=polar_p.tx.cfg.cfr_papr_db),
                         RFDAC(RFDACConfig(n_bits=14, iq_gain_db=G_DB,
                                           iq_phase_deg=p_deg)))
    e_meas.append(tx.run(wf, noise=False).evm().db)
    irr.append(-iq_image_rejection_db(G_DB, p_deg) if p_deg else np.nan)
a.plot(phis, e_meas, "s-", label=f"measured EVM (gain error {G_DB} dB, ideal 14-bit DAC)")
a.plot(phis, irr, "x--", label="-IRR closed form (image alone)")
a.set(xlabel="I/Q phase error [deg]", ylabel="EVM [dB]",
      title="the RF-DAC's sharpest knob: quadrature imbalance")
a.legend(fontsize=8)

# --------------------------- Part 4b: what the chain ignores, visibly
a = ax[1, 1]
skews_ns = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
e_polar = [wifi_dtc(bw=BW, qam=QAM, env_skew_s=s * 1e-9).tx.run(
    wf, noise=True, seed=SEED).evm().db for s in skews_ns]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    e_cart = [wifi_rfdac(bw=BW, qam=QAM, env_skew_s=s * 1e-9).tx.run(
        wf, noise=True, seed=SEED).evm().db for s in skews_ns]
a.plot(skews_ns, e_polar, "o-", label="polar: AM/PM path skew")
a.plot(skews_ns, e_cart, "^-", label="Cartesian: same knob, warned and ignored")
a.set(xlabel="envelope-path skew [ns]", ylabel="EVM [dB]",
      title=f"{sum(1 for w in caught if 'env_skew_s' in str(w.message))} "
            "UserWarnings raised: the knob is reported, not absorbed")
a.legend(fontsize=8)
fig.tight_layout()
path = os.path.join(OUT, "ex20_three_topologies.png")
fig.savefig(path, dpi=110)
print("saved", path)
