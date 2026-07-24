"""Example 12: supply pushing (the polar-specific AM->PM) and BLE
fractional channels.

Part 1  The class-D DPA's current draw follows the envelope; through
        the supply impedance and LO pushing that becomes envelope-
        correlated PM WITH MEMORY (the decoupling pole).  Static polar
        LUTs cannot fix it; the whole-chain Cartesian ILA partially
        can; the real fixes are supply rejection and pushing reduction.
Part 2  BLE channel sweep 2.402-2.480 GHz: fractional FCW through the
        TDC-mode ADPLL — EVM flat across the band.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx.cal.memory_dpd import fit_chain_ila, run_with_ila
from polartx.chain import ChainConfig, PolarTX, SupplyConfig
from polartx.dpa import DPA, DPAConfig
from polartx.phasemod import IdealPhaseModulator
from polartx.presets import ble_1m_adpll
from polartx.vendor.padpd.metrics import psd
from polartx.waveforms.ofdm import wifi_waveform

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

# ------------------------------------------------ Part 1: pushing law
wf = wifi_waveform(80e6, 256, n_symbols=4, seed=1)
print("=== supply pushing on WiFi-80 (R=0.15 ohm, tau=50 ns, I_fs=0.25 A) ===")
print(f"{'k_push [MHz/V]':>15s}{'phi rms [mrad]':>16s}{'EVM [dB]':>10s}")
kk = (0.0, 1e6, 2e6, 4e6, 8e6)
evm_k, phi_k = [], []
for k in kk:
    sup = SupplyConfig(k_push_hz_v=k) if k else None
    tx = PolarTX(ChainConfig(env_floor=0.02, supply=sup,
                             fs_scale_fixed=1.3 * np.abs(wf.x).max()),
                 IdealPhaseModulator(), DPA(DPAConfig(n_bits=11)))
    r = tx.run(wf, noise=False)
    evm_k.append(r.evm().db)
    phi_k.append(r.info.get("supply_phase_rms_mrad", 0.0))
    print(f"{k / 1e6:15.1f}{phi_k[-1]:16.2f}{evm_k[-1]:10.1f}")
a = ax[0]
a.plot([k / 1e6 for k in kk[1:]], evm_k[1:], "o-")
a.set(xlabel="LO pushing [MHz/V]", ylabel="EVM [dB]",
      title="supply-pushing law (6 dB per doubling)")
a.grid(alpha=0.3)

# ---------------------------- whole-chain ILA partially recovers it
tx = PolarTX(ChainConfig(env_floor=0.02, supply=SupplyConfig(k_push_hz_v=4e6),
                         fs_scale_fixed=1.3 * np.abs(wf.x).max()),
             IdealPhaseModulator(), DPA(DPAConfig(n_bits=11)))
r0 = tx.run(wf, noise=False)
dpd = fit_chain_ila(tx, wf)
r1 = run_with_ila(tx, wf, dpd, noise=False)
print(f"\nwhole-chain ILA on the pushing memory: EVM {r0.evm().db:.1f} -> "
      f"{r1.evm().db:.1f} dB (partial: GMP's |x| memory basis is not a "
      "supply-ripple model)")
a = ax[1]
for res, lab, c in ((r0, "pushing, no DPD", "tab:red"),
                    (r1, "with whole-chain ILA", "tab:green")):
    f, pdb = psd(res.y, res.fs, nfft=8192)
    a.plot(f / 1e6, pdb, lw=0.7, color=c, label=lab)
a.set(xlim=(-320, 320), ylim=(-90, 5), xlabel="offset [MHz]",
      ylabel="dBr", title="pushing spectrum and ILA recovery")
a.legend(fontsize=8)

# -------------------------------- Part 2: BLE fractional channel sweep
print("\n=== BLE channel sweep (fractional FCW, fref = 32 MHz) ===")
chans = [2.402e9, 2.420e9, 2.440e9, 2.462e9, 2.479e9]
evm_c = []
wfb = None
for fout in chans:
    p = ble_1m_adpll(fout=fout)
    wfb = p.make_waveform(n_bits=400, seed=5)
    evm_c.append(p.tx.run(wfb, noise=True, seed=3).evm()["evm_pct"])
    print(f"  {fout / 1e9:.3f} GHz (FCW {fout / 32e6:.4f}): "
          f"EVM {evm_c[-1]:.2f}%")
a = ax[2]
a.plot([c / 1e9 for c in chans], evm_c, "o-")
a.set(xlabel="channel [GHz]", ylabel="EVM [%]",
      title="BLE band: EVM flat over fractional channels")
a.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex12_supply_pushing.png"), dpi=130)
print(f"\nplots -> {OUT}/ex12_supply_pushing.png")
