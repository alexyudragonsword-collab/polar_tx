"""Example 16: architecture selector — four transmitter topologies.

Given a requirement (standard, bandwidth, EVM target), rank the four
topologies and recommend one, with a first-order EVM budget and an average
efficiency from each topology's own law:

  * open-loop DTC (polar) = shared synth PN (+) DTC quant/jitter/INL
                     floors, feasible at any bandwidth
  * ADPLL two-point (polar) = shared synth PN (+) two-point mismatch,
                     in-loop FM (no DTC floors) but infeasible past the
                     direct-DAC coverage ceiling
  * outphasing     = the DTC floors, each +10log10(PAPR/2) dB because both
                     branches run at full scale, (+) branch phase mismatch
  * Cartesian RF-DAC = shared synth PN (+) I/Q quantization over the
                     oversampling ratio (+) LO jitter (+) the I/Q image;
                     no phase path, the |I|+|Q| efficiency law.

Ranking: feasible, then meeting the target, then the HIGHEST efficiency
(margin beyond the target buys nothing), then the lowest EVM.

Part 1  Decision table for the standards this library targets.
Part 2  EVM-vs-bandwidth crossover chart: the calibrated ADPLL wins across
        narrowband up to its coverage ceiling; beyond it the three
        bandwidth-agnostic topologies compete on floors and efficiency.
        The uncalibrated ADPLL curve shows what the online two-point
        calibration is worth.

Scores are analytic (confirm with the real chain via the suggested preset;
ex20 runs the three wideband topologies on one burst).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polartx.selector import Requirement, select

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def part1_table():
    cases = [
        Requirement("BLE-1M",    1e6,   "gfsk", evm_db_max=-20,
                    constant_envelope=True, fout=2.44e9),
        Requirement("LTE-20",    18e6,  "ofdm", evm_db_max=-25, fout=1.95e9),
        Requirement("WiFi6-80",  80e6,  "ofdm", evm_db_max=-38, fout=5.8e9,
                    dtc_bits=11),
        Requirement("WiFi7-320", 320e6, "ofdm", evm_db_max=-38, fout=6e9,
                    dtc_bits=12),
        Requirement("NR-200",    200e6, "ofdm", evm_db_max=-30, fout=3.5e9,
                    dtc_bits=12),
    ]
    print("=" * 74)
    print("Part 1 — architecture decision table")
    print("=" * 74)
    for req in cases:
        rep = select(req)
        print(f"\n### {req.standard}  ({req.bw_hz/1e6:.0f} MHz, "
              f"target {req.evm_db_max:.0f} dB)")
        print(rep.table())
        print("  ->", rep.recommendation)
        print("     closest preset:", rep.suggest_preset())


def part2_crossover():
    bws = np.logspace(np.log10(1e6), np.log10(320e6), 40)
    adpll_cal, adpll_unc, dtc, outph, rfdac = [], [], [], [], []
    for bw in bws:
        rc = select(Requirement("c", bw, "ofdm", fout=3.5e9,
                                two_point_gain_match=2e-3))
        ru = select(Requirement("u", bw, "ofdm", fout=3.5e9,
                                two_point_gain_match=5e-3))
        a_c = next(c for c in rc.candidates if c.arch == "adpll_two_point")
        a_u = next(c for c in ru.candidates if c.arch == "adpll_two_point")
        d = next(c for c in rc.candidates if c.arch == "dtc_open_loop")
        adpll_cal.append(a_c.evm_db if a_c.feasible else np.nan)
        adpll_unc.append(a_u.evm_db if a_u.feasible else np.nan)
        dtc.append(d.evm_db)
        outph.append(next(c for c in rc.candidates if c.arch == "outphasing").evm_db)
        rfdac.append(next(c for c in rc.candidates if c.arch == "rfdac_cartesian").evm_db)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(bws / 1e6, adpll_cal, "-o", ms=3, label="ADPLL two-point (0.2% cal)")
    ax.semilogx(bws / 1e6, adpll_unc, "--s", ms=3, color="C0", alpha=0.5,
                label="ADPLL two-point (0.5% uncal)")
    ax.semilogx(bws / 1e6, dtc, "-^", ms=3, label="open-loop DTC")
    ax.semilogx(bws / 1e6, outph, "-s", ms=3, label="outphasing (2 DTC branches, 1 deg mismatch)")
    ax.semilogx(bws / 1e6, rfdac, "-d", ms=3, label="Cartesian RF-DAC (12 bit, IRR 46 dB)")
    ceil = 50.0
    ax.axvspan(ceil, bws[-1] / 1e6, color="red", alpha=0.06)
    ax.axvline(ceil, color="red", ls=":", lw=1)
    ax.text(ceil * 1.05, ax.get_ylim()[1] - 2, "ADPLL two-point\ninfeasible",
            color="red", fontsize=8, va="top")
    ax.set_xlabel("signal bandwidth (MHz)")
    ax.set_ylabel("estimated EVM (dB)")
    ax.set_title("Transmitter topology crossover, analytic (fout = 3.5 GHz)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "ex16_architecture_crossover.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    part1_table()
    part2_crossover()
