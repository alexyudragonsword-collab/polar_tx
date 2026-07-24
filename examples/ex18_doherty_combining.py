"""Example 18: multi-core / Doherty DPA power combining.

The single-core DPA carries a fitted Doherty efficiency law; this models the
combining itself and DERIVES both outputs from the two-core load modulation.

Part 1  Efficiency vs backoff: ideal class-B flat top vs class-C double-hump,
        2-way vs 3-way extended Doherty — and the average efficiency a real
        OFDM envelope would see, versus a plain single-core SCPA.
Part 2  Core imbalance: gain/phase mismatch between the main and peaking cores
        turns into an AM-AM/AM-PM handoff kink; Monte-Carlo the matching to a
        yield spec.
Part 3  Drop the combiner into a DPA and run the WiFi chain: EVM and average
        efficiency of a Doherty-DPA vs the single-core baseline.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx.dpa import (DPA, DPAConfig, DohertyCombiner, imbalance_montecarlo)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def part1_efficiency():
    x = np.linspace(0.02, 1.0, 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    variants = [
        ("single-core SCPA", DPAConfig(eff=("scpa", 0.67, 0.85)), "C3"),
        ("2-way Doherty (ideal B)", DohertyCombiner(n_way=2, backoff_db=6.0,
                                                    eta_peak=0.85, combiner_loss_db=0.0), "C0"),
        ("2-way Doherty (class-C)", DohertyCombiner(n_way=2, backoff_db=6.0,
                                                    eta_peak=0.85, peaking="C",
                                                    combiner_loss_db=0.4), "C1"),
        ("3-way ext. Doherty", DohertyCombiner(n_way=3, backoff_db=9.5,
                                               eta_peak=0.85, peaking="C",
                                               combiner_loss_db=0.4), "C2"),
    ]
    bo_db = -20 * np.log10(x)
    for name, obj, col in variants:
        if isinstance(obj, DPAConfig):
            from polartx.dpa.characteristics import efficiency_curve
            eta = efficiency_curve(obj.eff, x)
        else:
            eta = obj.efficiency(x)
        ax.plot(bo_db, 100 * eta, color=col, label=name)
    ax.set_xlim(18, 0)
    ax.set_xlabel("output backoff (dB)")
    ax.set_ylabel("drain efficiency (%)")
    ax.set_title("Doherty power combining: derived efficiency vs backoff")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "ex18_doherty_efficiency.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print("Part 1 — efficiency")
    for name, obj, _ in variants[1:]:
        # average efficiency over a Rayleigh-ish OFDM envelope at 8 dB PAPR
        rng = np.random.default_rng(0)
        env = np.abs(rng.normal(size=200000) + 1j * rng.normal(size=200000))
        env = env / np.percentile(env, 99.9)
        env = np.clip(env, 0, 1)
        eta = obj.efficiency(env)
        p_out = env ** 2
        on = eta > 0
        eta_avg = p_out[on].sum() / (p_out[on] / eta[on]).sum()
        print(f"  {name:26s}: avg eff over OFDM env = {100*eta_avg:.1f}%")
    print(f"  saved {p}")


def part2_imbalance():
    print("\nPart 2 — core imbalance")
    clean = DohertyCombiner(n_way=2, backoff_db=6.0)
    dirty = DohertyCombiner(n_way=2, backoff_db=6.0,
                            gain_imbalance=(0.0, 0.12),
                            phase_imbalance_deg=(0.0, 10.0))
    cc, cd = clean.am_curves(), dirty.am_curves()
    print(f"  clean : AM-AM ripple {cc['amam_ripple_db']:.3f} dB, "
          f"AM-PM {cc['ampm_pp_deg']:.2f} deg")
    print(f"  dirty : AM-AM ripple {cd['amam_ripple_db']:.3f} dB, "
          f"AM-PM {cd['ampm_pp_deg']:.2f} deg, "
          f"combining loss {dirty.combining_loss_db():.2f} dB")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(cc["x"], 20 * np.log10(np.clip(cc["amam"], 1e-6, None) /
            np.clip(cc["x"], 1e-6, None)), label="clean")
    a1.plot(cd["x"], 20 * np.log10(np.clip(cd["amam"], 1e-6, None) /
            np.clip(cd["x"], 1e-6, None)), label="imbalanced")
    a1.axvline(10 ** (-6 / 20.0), color="k", ls=":", lw=1, label="handoff")
    a1.set_xlabel("input amplitude"); a1.set_ylabel("AM-AM gain error (dB)")
    a1.set_title("Doherty handoff kink"); a1.grid(True, alpha=0.3); a1.legend(fontsize=8)
    a2.plot(cd["x"], np.rad2deg(cd["ampm_rad"]))
    a2.axvline(10 ** (-6 / 20.0), color="k", ls=":", lw=1)
    a2.set_xlabel("input amplitude"); a2.set_ylabel("AM-PM (deg)")
    a2.set_title("AM-PM from core phase imbalance"); a2.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, "ex18_doherty_imbalance.png")
    fig.savefig(p, dpi=110); plt.close(fig)

    mc = imbalance_montecarlo(clean, sigma_gain=0.02, sigma_phase_deg=3.0,
                              n_trials=500)
    print(f"  MC (2% gain, 3deg phase, {mc['n_trials']} trials):")
    print(f"    AM-AM ripple p95 = {mc['amam_ripple_db']['p95']:.3f} dB")
    print(f"    AM-PM pp    p95 = {mc['ampm_pp_deg']['p95']:.2f} deg")
    print(f"    combining loss p05 = {mc['combining_loss_db']['p05']:.2f} dB")
    print(f"  saved {p}")


def part3_chain():
    print("\nPart 3 — Doherty-DPA in the WiFi chain")
    from polartx.presets import wifi_dtc
    specs = DohertyCombiner(n_way=2, backoff_db=6.0, eta_peak=0.85,
                            peaking="C", combiner_loss_db=0.4).to_dpa_specs()
    for label, dpacfg in [
        ("single-core", DPAConfig(n_bits=11, amam=("rapp", 2.5, 1.1),
                                  ampm_deg_poly=(0.0, 2.0),
                                  eff=("scpa", 0.67, 0.85))),
        ("Doherty combined", DPAConfig(n_bits=11, amam=specs["amam"],
                                       ampm_lut=specs["ampm_lut"],
                                       eff=specs["eff"]))]:
        p = wifi_dtc(bw=80e6, qam=1024, dpa=dpacfg, dpd=True)
        wf = p.make_waveform(n_symbols=6, seed=0)
        res = p.tx.run(wf)
        eta = res.avg_efficiency(p.tx.dpa)["eta_avg"]
        print(f"  {label:18s}: EVM {res.evm().db:6.1f} dB, "
              f"avg eff {100*eta:.1f}%")


if __name__ == "__main__":
    part1_efficiency()
    part2_imbalance()
    part3_chain()
