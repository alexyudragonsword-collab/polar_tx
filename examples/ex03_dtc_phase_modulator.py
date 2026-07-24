"""Example 03: the wideband open-loop DTC phase modulator, standalone.

Design charts for the phase path of the WiFi/NR polar TX: quantization
floor vs resolution (with and without first-order dither), INL spurs vs
the closed-form prediction, ZOH images from a slow phase-update clock,
and the locked-LO phase-noise contribution — each simulated level checked
against its analytic companion (polartx.analysis.responses).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx.analysis.responses import (dtc_quant_phase_rms,
                                        evm_db_from_phase_rms,
                                        inl_sin_spur_dbc, zoh_image_dbc)
from polartx.phasemod import DTCPhaseModulator, DTCPMConfig
from polartx.polar import polar_split
from polartx.vendor.padpd.metrics import evm_of_signal, psd
from polartx.waveforms.ofdm import wifi_waveform

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
TWOPI = 2 * np.pi

wf = wifi_waveform(160e6, 4096, n_symbols=4, seed=3)
env, phase, _ = polar_split(wf.x, env_floor=0.02)
fs = wf.fs

# ------------------------------------------------ EVM vs DTC resolution
print("=== phase-quantization EVM vs DTC bits (160 MHz 4096-QAM phase) ===")
print(f"{'bits':>5s}{'EVM sim':>10s}{'EVM sim+dith':>14s}{'analytic':>10s}")
bits_list = range(6, 13)
evm_q, evm_d = [], []
for bits in bits_list:
    row = []
    for dith in (False, True):
        pm = DTCPhaseModulator(DTCPMConfig(n_bits=bits, dither=dith))
        out = pm.modulate(phase, fs, noise=False).phase_out
        y = env * np.exp(1j * out)
        row.append(evm_of_signal(y, wf.ofdm_ref).db)
    evm_q.append(row[0])
    evm_d.append(row[1])
    ana = evm_db_from_phase_rms(dtc_quant_phase_rms(bits, osr=fs / wf.bw))
    print(f"{bits:5d}{row[0]:9.1f} {row[1]:13.1f} {ana:9.1f}")
print("4096-QAM (-38 dB) needs ~10 bits plain, ~9 with dither\n")

# --------------------------------------------------------- INL spurs
amp_ui, k, f_off = 2e-3, 3, 5e6
n = 1 << 16
cw = TWOPI * f_off * np.arange(n) / fs
pm = DTCPhaseModulator(DTCPMConfig(n_bits=14, inl_sin=(amp_ui, k, 0.0)))
out = pm.modulate(cw, fs, noise=False).phase_out
print(f"INL sin ({amp_ui * 1e3:.0f} mUI, {k} cycles), CW at {f_off / 1e6:.0f} MHz: "
      f"predicted spur {inl_sin_spur_dbc(amp_ui):.1f} dBc at k*f_off")
print(f"ZOH: phase update fs/4 -> image at f_sig-f_update, predicted "
      f"{zoh_image_dbc(10e6, fs / 4):.1f} dBc\n")

# ------------------------------------------------------------------ plots
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

a = ax[0]
a.plot(list(bits_list), evm_q, "o-", label="round quantizer")
a.plot(list(bits_list), evm_d, "s-", label="+ 1st-order dither")
a.plot(list(bits_list),
       [evm_db_from_phase_rms(dtc_quant_phase_rms(b, osr=fs / wf.bw))
        for b in bits_list], "k--", lw=1, label="analytic floor")
a.axhline(-38, color="r", ls=":", label="4096-QAM req")
a.set(xlabel="DTC bits (per 2π)", ylabel="EVM [dB]",
      title="DTC resolution design chart")
a.grid(True, alpha=0.3)
a.legend()

a = ax[1]
spec = np.abs(np.fft.fft(np.exp(1j * out) * np.hanning(n)))
db = 20 * np.log10(spec / spec.max())
fx = np.fft.fftfreq(n, 1 / fs)
idx = np.argsort(fx)
a.plot(fx[idx] / 1e6, db[idx], lw=0.6)
a.axhline(inl_sin_spur_dbc(amp_ui), color="r", ls="--",
          label="predicted INL spur")
a.set(xlabel="frequency [MHz]", ylabel="dBc", xlim=(-40, 40),
      ylim=(-100, 5), title=f"INL spurs, {k} cycles x {f_off / 1e6:.0f} MHz CW")
a.legend()

a = ax[2]
from polartx.presets import wifi_dtc
pm_lo = DTCPhaseModulator(wifi_dtc(bw=160e6).tx.phasemod.cfg)
out_lo = pm_lo.modulate(phase, fs, noise=True, seed=2).phase_out
y = env * np.exp(1j * out_lo)
f, p = psd(y, fs, nfft=8192)
a.plot(f / 1e6, p, lw=0.7, label="TX out (locked LO PN)")
f, p = psd(env * np.exp(1j * phase), fs, nfft=8192)
a.plot(f / 1e6, p, lw=0.7, label="ideal")
a.set(xlabel="frequency [MHz]", ylabel="dBr", ylim=(-90, 5),
      title=f"LO PN contribution (EVM {evm_of_signal(y, wf.ofdm_ref).db:.1f} dB)")
a.legend()

fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex03_dtc_phase_modulator.png"), dpi=130)
print(f"plots -> {OUT}/ex03_dtc_phase_modulator.png")
