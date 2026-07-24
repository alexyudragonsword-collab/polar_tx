"""Example 05: the pi-flip problem — DEVM vs ACP under a finite
direct-modulation DAC, and trajectory-side slew limiting.

8DPSK envelopes cross zero, so the raw polar phase command flips by pi in
one sample: fs/2 = 16 MHz of instantaneous deviation the ADPLL's direct
DAC must cover.  DEVM alone hides this (it samples symbol centers, and
the DPA output is tiny at the null) — but the moment the DAC range is
finite, the clipped phase error is recovered only through the loop's
lowpass path (~loop-BW timescale, tens of symbols), and DEVM AND ACP
collapse together.  Slew-limiting the trajectory at the polar split
(vector hole punching: linearize the phase across each null) bounds the
required range at an exactly-computable trajectory-EVM cost.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx.chain import ChainConfig
from polartx.metrics import check_mask
from polartx.metrics.ble_metrics import bt_acp
from polartx.metrics.masks import ble_mask
from polartx.presets import bt_edr_adpll
from polartx.vendor.padpd.metrics import psd

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

wf = bt_edr_adpll("8dpsk").make_waveform(n_syms=600, seed=2)


def run(dp_range=None, slew=None):
    ch = ChainConfig(env_floor=0.05, phase_slew_max_hz=slew)
    p = bt_edr_adpll("8dpsk", mode="event", dp_range_hz=dp_range, chain=ch)
    r = p.tx.run(wf, noise=True, seed=3)
    acp = bt_acp(r.y, r.fs)
    return r, {
        "devm_pct": r.evm()["devm_pct"],
        "acp2": max(acp["acp+2MHz_dbc"], acp["acp-2MHz_dbc"]),
        "acp3": max(acp["acp+3MHz_dbc"], acp["acp-3MHz_dbc"]),
        "req_mhz": r.info["phasemod"]["dp_required_range_hz"] / 1e6,
        "clip": r.info["phasemod"]["dp_clip_frac"],
        "mod_evm": r.info["split"]["mod_evm_db"],
    }


print("=== 8DPSK: direct-DAC range x trajectory slew limit ===")
print(f"{'slew lim':>9s}{'dp range':>9s}{'req rng':>9s}{'clip':>7s}"
      f"{'traj EVM':>10s}{'DEVM':>7s}{'ACP2M':>8s}{'ACP3M':>8s}")
cases = [(None, None), (None, 8e6), (None, 4e6), (None, 2e6),
         (2e6, None), (2e6, 2e6)]
rows = {}
for slew, rng in cases:
    r, m = run(rng, slew)
    rows[(slew, rng)] = (r, m)
    s = "none" if slew is None else f"{slew / 1e6:.0f}M"
    g = "unlim" if rng is None else f"{rng / 1e6:.0f}M"
    mod = "  --  " if not np.isfinite(m["mod_evm"]) else f"{m['mod_evm']:5.1f}dB"
    print(f"{s:>9s}{g:>9s}{m['req_mhz']:8.1f}M{100 * m['clip']:6.1f}%"
          f"{mod:>10s}{m['devm_pct']:6.2f}%{m['acp2']:8.1f}{m['acp3']:8.1f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

# DEVM & ACP vs DAC range (raw trajectory)
rngs = (16e6, 8e6, 4e6, 2e6)
devm_r, acp_r = [], []
for rng in rngs:
    _, m = run(rng, None)
    devm_r.append(m["devm_pct"])
    acp_r.append(m["acp2"])
a = ax[0]
a.plot([r / 1e6 for r in rngs], devm_r, "o-", color="tab:red")
a.axhline(13, ls=":", color="k", label="EDR3 DEVM limit 13%")
a.set(xlabel="direct-DAC range [MHz]", ylabel="DEVM [%]",
      title="raw trajectory: DEVM collapses\n(loop-BW hangover after each clip)")
a2 = a.twinx()
a2.plot([r / 1e6 for r in rngs], acp_r, "s--", color="tab:orange")
a2.set_ylabel("ACP@2MHz [dBc]", color="tab:orange")
a.legend()
a.grid(alpha=0.3)

# spectra comparison
a = ax[1]
for key, lab, c in (((None, None), "unlimited DAC", "tab:blue"),
                    ((None, 2e6), "2 MHz DAC, raw traj", "tab:red"),
                    ((2e6, 2e6), "2 MHz DAC + 2 MHz slew-limited traj",
                     "tab:green")):
    r, m = rows[key]
    f, p = psd(r.y, r.fs, nfft=8192)
    a.plot(f / 1e6, p, lw=0.7, color=c, label=lab)
mask = ble_mask(1e6)
f0 = np.linspace(-5e6, 5e6, 400)
a.plot(f0 / 1e6, np.interp(np.abs(f0), mask[:, 0], mask[:, 1]), "k--",
       lw=1, label="BT mask")
a.set(xlim=(-5, 5), ylim=(-80, 5), xlabel="offset [MHz]", ylabel="dBr",
      title="ACP is the sensitive metric,\nthe coarse mask is not")
a.legend(fontsize=8)

# phase trajectory around a null: raw vs slew-limited
from polartx.polar import polar_split
_, ph_raw, _ = polar_split(wf.x, env_floor=0.05)
_, ph_lim, _ = polar_split(wf.x, env_floor=0.05, phase_slew_max_hz=2e6,
                           fs=wf.fs)
k = int(np.argmax(np.abs(np.diff(ph_raw))))
t = np.arange(-40, 40)
a = ax[2]
a.plot(t, ph_raw[k - 40: k + 40], label="raw phase (pi flip)")
a.plot(t, ph_lim[k - 40: k + 40], label="slew-limited (2 MHz)")
a.set(xlabel="sample", ylabel="phase [rad]",
      title="vector hole punching at a null")
a.legend()
a.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex05_edr_pi_jump.png"), dpi=130)
print(f"\nplots -> {OUT}/ex05_edr_pi_jump.png")
