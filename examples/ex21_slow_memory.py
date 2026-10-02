"""Example 21: slow-state (thermal) memory from a complete measured source.

Stage 2 of inverting measured memory into the polar chain.  ex09 handles
FAST memory (GMP-class, tens of samples) from a stationary capture.  Slow
memory — the gain following the device's recent envelope POWER over
microseconds (self-heating, bias, trapping) — is invisible in a stationary
capture, so it comes from a padpd "complete source": a ``step`` group
(constant-envelope step probe) that identifies the time constants offline,
and a ``burst`` group (power-stepped OFDM) that trains a
StateConditionedSpline residual with exactly those constants.

No hardware here: the source is recorded from a VIRTUAL self-heating DUT
(vendored ThermalReferencePA, true taus 5 / 30 us), written to the file
format and read back, as a recorded one would be.  Swap in your own file
with ``dpa_from_complete_source("my_capture.npz")``.

Part 1  tau identification from the step capture alone.
Part 2  the residual ladder on the burst: static LUT / + memoryless /
        + fast-only SplineGMP / + slow states, and the same models on
        the stationary main capture, where slow state must buy nothing.
Part 3  the chain: measured DPA + state memory, driven with the burst,
        against the measured output.
Part 4  why tau is identified, not guessed: state gain vs a tau error.
"""
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx.chain import ChainConfig, PolarTX
from polartx.measured import (SYNTHETIC_THERMAL_TAUS_S, _fit_on_pair,
                              dpa_from_complete_source, fast_spline_residual,
                              identify_slow_states, residual_training_pair,
                              state_spline_residual, static_prediction,
                              synthetic_thermal_source)
from polartx.phasemod import IdealPhaseModulator
from polartx.vendor.padpd.pa import nmse_db
from polartx.waveforms.base import Waveform

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

path = os.path.join(tempfile.mkdtemp(), "thermal_source.npz")
src = synthetic_thermal_source(path)
fs = src["fs"]
print(f"complete source {os.path.basename(path)}: fs {fs / 1e6:.0f} MS/s, groups "
      f"{sorted(src['extras'])}, main capture: {src['meta']['main_capture']}")

# ------------------------------------------------ Part 1: tau from the step
gm = identify_slow_states(src)
print("\n=== Part 1: offline tau identification (step capture) ===")
for got, true in zip(gm.taus_heat_s, SYNTHETIC_THERMAL_TAUS_S):
    print(f"  heating tau {got * 1e6:6.2f} us   (DUT truth {true * 1e6:.0f} us, "
          f"error {100 * (got - true) / true:+.1f} %)")
print(f"  cooling tau {', '.join(f'{t * 1e6:.2f} us' for t in gm.taus_cool_s)}; "
      f"gain droop on the step {gm.droop_db:.2f} dB")

# --------------------------------------------- Part 2: the residual ladder
dpa, ch = dpa_from_complete_source(src)
rep = ch["memory_report"]
alphas = rep["state_alphas"]
u_m, v_m = residual_training_pair(ch)
st_m = _fit_on_pair(u_m, v_m, lambda ut: state_spline_residual(ut, alphas), 0.6, 1e-9, ch)[1]
fa_m = _fit_on_pair(u_m, v_m, fast_spline_residual, 0.6, 1e-9, ch)[1]
print("\n=== Part 2: residual NMSE on the validation tail (chain scale, dB) ===")
hdr = f"{'capture':24s} {'static LUT':>11s} {'+memoryless':>12s} {'+fast only':>11s} {'+slow state':>12s}"
print(hdr)
print("-" * len(hdr))
print(f"{'burst (power-stepped)':24s} {rep['static_nmse_db']:11.2f} {rep['memoryless_nmse_db']:12.2f} "
      f"{rep['fast_only_nmse_db']:11.2f} {rep['residual_nmse_db']:12.2f}")
print(f"{'main (warm, stationary)':24s} {st_m['static_nmse_db']:11.2f} {st_m['memoryless_nmse_db']:12.2f} "
      f"{fa_m['residual_nmse_db']:11.2f} {st_m['residual_nmse_db']:12.2f}")
print(f"slow state buys {rep['state_gain_db']:.1f} dB on the burst and "
      f"{fa_m['residual_nmse_db'] - st_m['residual_nmse_db']:+.2f} dB on the stationary capture")
print(f"state model: {rep['n_coeffs']} coefficients, rank deficiency {rep['rank_deficiency']} "
      f"({rep['dead_columns']} dead columns), condition {rep['condition_number']:.1e} on the spanned space")
from polartx.vendor.padpd.pa.thermal import ThermalReferencePA
_pa = ThermalReferencePA(drive0=0.13, fs=fs, heat_gain=0.0)
_floor = nmse_db(_pa._drift.pa()(src["x"]), _pa(src["x"]))
print(f"virtual-DUT floor: {_floor:.1f} dB -- its FIRs restart every 128-sample block, a\n"
      f"non-physical glitch no model can fit; the stationary residual sits on it, so\n"
      f"these NMSEs are DUT-limited and the state gain is a lower bound")

# ------------------------------------------------------- Part 3: the chain
b = src["extras"]["burst"]
g = ch["chain_gain"]
wf = Waveform(x=b["x"], fs=fs, bw=fs / 4, kind="ofdm")
cfg = ChainConfig(fs_scale_fixed=ch["fs_scale"])
val = slice(rep["n_train"], None)
y_static = PolarTX(cfg, IdealPhaseModulator(), dpa).run(wf, noise=False).y
y_mem = PolarTX(cfg, IdealPhaseModulator(), dpa, memory=ch["memory"]).run(wf, noise=False).y
v_b = b["y"] / g
print("\n=== Part 3: polar chain driven with the burst input (validation tail) ===")
print(f"  measured DPA, no memory           NMSE vs measured {nmse_db(v_b[val], y_static[val]):7.2f} dB")
print(f"  measured DPA + slow-state memory  NMSE vs measured {nmse_db(v_b[val], y_mem[val]):7.2f} dB")

# --------------------------------------------- Part 4: tau error sensitivity
u_b = static_prediction(ch, b["x"]) / g
scales = np.array([0.03, 0.1, 0.3, 0.6, 1.0, 1.6, 3.0, 10.0])
fast_b = rep["fast_only_nmse_db"]
gains = []
for s in scales:
    al = np.exp(-1.0 / (np.asarray(gm.taus_heat_s) * s * fs))
    r = _fit_on_pair(u_b, v_b, lambda ut, al=al: state_spline_residual(ut, al), 0.6, 1e-9, ch)[1]
    gains.append(fast_b - r["residual_nmse_db"])
print("\n=== Part 4: state gain vs a tau error (x identified tau) ===")
print("  " + "  ".join(f"x{s:g}: {gn:+.1f}" for s, gn in zip(scales, gains)) + "  dB")

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
a = ax[0, 0]
xs, ys = src["extras"]["step"]["x"], src["extras"]["step"]["y"]
t_us = np.arange(xs.size) / fs * 1e6
gain_db = 20 * np.log10(np.abs(ys) / np.abs(xs))
# the virtual DUT restarts its FIRs every 128-sample block (an upstream
# artefact, see Part 2's floor): drop the first three samples of each block
clean = (np.arange(xs.size) % 128) >= 3
gain_db = gain_db - np.median(gain_db[clean][-1000:])
a.plot(t_us[clean], gain_db[clean], lw=0.8)
a.set(xlabel="time [us]", ylabel="gain re. final cold state [dB]",
      title=f"step probe: identified tau {gm.taus_heat_s[0] * 1e6:.1f} / "
            f"{gm.taus_heat_s[1] * 1e6:.1f} us (truth 5 / 30)")
a.grid(alpha=0.3)

a = ax[0, 1]
fast_model, _ = _fit_on_pair(u_b, v_b, fast_spline_residual, 0.6, 1e-9, ch)
win = 400
kernel = np.ones(win) / win
tb = np.arange(u_b.size) / fs * 1e6
for lab, yhat in (("fast-only SplineGMP", fast_model(u_b)), ("+ slow states", ch["memory"](u_b))):
    e = np.convolve(np.abs(v_b - yhat) ** 2, kernel, "same")
    a.semilogy(tb, np.sqrt(e), lw=1.0, label=lab)
a2 = a.twinx()
a2.plot(tb, np.convolve(np.abs(b["x"]) ** 2, kernel, "same"), color="grey", lw=0.6, alpha=0.5)
a2.set_ylabel("drive power (grey)")
a.axvspan(tb[rep["n_train"]], tb[-1], color="C2", alpha=0.08, label="validation")
a.set(xlabel="time [us]", ylabel="rolling rms error",
      title="burst: the stateless model errs after every power step")
a.legend(fontsize=8, loc="upper left")

a = ax[1, 0]
labels = ["static LUT", "+memoryless", "+fast only", "+slow state"]
burst_v = [rep["static_nmse_db"], rep["memoryless_nmse_db"], rep["fast_only_nmse_db"], rep["residual_nmse_db"]]
main_v = [st_m["static_nmse_db"], st_m["memoryless_nmse_db"], fa_m["residual_nmse_db"], st_m["residual_nmse_db"]]
xi = np.arange(len(labels))
a.bar(xi - 0.2, burst_v, 0.4, label="burst (power-stepped)")
a.bar(xi + 0.2, main_v, 0.4, label="main (warm, stationary)")
a.set(xticks=xi, xticklabels=labels, ylabel="NMSE [dB]", title="residual ladder: slow state only pays where it is observable")
a.invert_yaxis()
a.legend(fontsize=8)
a.grid(alpha=0.3, axis="y")

a = ax[1, 1]
a.semilogx(scales, gains, "o-")
a.axhline(0, color="k", lw=0.8)
a.axvline(1.0, color="C2", ls="--", lw=0.8)
a.set(xlabel="tau used / tau identified", ylabel="state gain over fast-only [dB]",
      title="a wrong tau is not neutral: identify it, don't guess")
a.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex21_slow_memory.png"), dpi=120)
print(f"\nplots -> {OUT}/ex21_slow_memory.png")
