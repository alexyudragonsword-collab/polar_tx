"""Example 13: full EDR packet timing and burst power ramping.

Part 1  A real EDR burst: GFSK access-code/header (constant envelope)
        -> guard -> SRRC 8DPSK payload, through the ADPLL polar chain
        in one run; header delta-f and payload DEVM measured on their
        own segments.
Part 2  Burst ramping: ramp specs are MAX-HOLD power-vs-time in the
        adjacent channel.  A Welch average cannot even see the keying
        transient (its taper kills the edges); the max-hold detector
        shows a hard-keyed burst at -20 dBc vs -56 dBc with a 2 us
        raised-cosine ramp — the reason every TX ramps.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx.impairments import apply_ramp
from polartx.metrics.ble_metrics import acp_transient_db
from polartx.metrics.dpsk import packet_metrics
from polartx.presets import bt_edr_adpll
from polartx.waveforms.edr import edr_packet

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

# ------------------------------------------------ Part 1: full packet
p = bt_edr_adpll("8dpsk")
wf = edr_packet(400, 32e6, mode="8dpsk", seed=2)
res = p.tx.run(wf, noise=True, seed=3)
m = packet_metrics(res.y, wf)
print("=== EDR full packet through the ADPLL polar chain ===")
print(f"header GFSK delta-f avg : {m['header_dev_avg_hz'] / 1e3:.1f} kHz")
print(f"payload 8DPSK DEVM      : {m['payload_devm_pct']:.2f} % (limit 13%)")

a = ax[0]
t = np.arange(wf.n) / wf.fs * 1e6
a.plot(t, np.abs(res.y) / np.abs(res.y).max(), lw=0.5)
for name, (s0, s1) in wf.meta["segments"].items():
    a.axvspan(t[s0], t[min(s1, wf.n - 1)], alpha=0.08,
              color={"gfsk": "tab:blue", "guard": "tab:orange",
                     "dpsk": "tab:green"}[name])
    a.text(t[(s0 + s1) // 2], 1.06, name, ha="center", fontsize=9)
a.set(xlabel="time [us]", ylabel="|y| (norm)", ylim=(0, 1.15),
      title="packet envelope: GFSK -> guard -> 8DPSK")

# ------------------------------------------------ Part 2: ramp study
wfb = p.make_waveform(n_syms=200, seed=2)
y0 = p.tx.run(wfb, noise=False).y
ramps = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0]) * 1e-6
tr = [acp_transient_db(apply_ramp(y0, wfb.fs, r), wfb.fs) for r in ramps]
print("\n=== burst ramp: transient ACP@2MHz (max-hold) ===")
for r, v in zip(ramps, tr):
    print(f"  ramp {r * 1e6:4.2f} us: {v:6.1f} dBc")

a = ax[1]
a.plot(ramps * 1e6, tr, "o-")
a.axhline(-40, ls=":", color="r", label="stylized transient limit")
a.set(xlabel="ramp time [us]", ylabel="transient ACP@2MHz [dBc]",
      title="ramp-time design chart (max-hold)")
a.grid(alpha=0.3)
a.legend()

a = ax[2]
for r, c in ((0.0, "tab:red"), (2e-6, "tab:green")):
    y = apply_ramp(y0, wfb.fs, r)
    n = y.size
    z = y * np.exp(-2j * np.pi * 2e6 * np.arange(n) / wfb.fs)
    spec = np.fft.fft(z)
    f = np.fft.fftfreq(n, 1 / wfb.fs)
    spec[np.abs(f) > 0.5e6] = 0
    env = np.abs(np.fft.ifft(spec)) ** 2
    p_in = np.mean(np.abs(y) ** 2)
    a.plot(np.arange(n) / wfb.fs * 1e6, 10 * np.log10(env / p_in + 1e-12),
           lw=0.6, color=c,
           label=f"ramp {r * 1e6:.0f} us" if r else "hard keyed")
a.set(xlabel="time [us]", ylabel="adjacent-ch power [dBc]",
      ylim=(-80, -10), title="power-vs-time in the +2 MHz channel")
a.legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex13_packet_and_ramp.png"), dpi=130)
print(f"\nplots -> {OUT}/ex13_packet_and_ramp.png")
