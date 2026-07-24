"""Example 17: fuller RTL/AMS export of the digital polar-TX datapath.

Beyond the polar-DPD dual LUT (ex/ tests already cover it), this emits the
rest of the synthesizable digital datapath plus the analog PA bridge, each
bit-true-verified against its Python golden with iverilog where available:

  CFR clip        polar envelope peak limiter (compare + mux)
  DTC phase acc   modulo-2^w phase accumulator (open-loop phase datapath)
  DPA decoder     segmented binary -> thermometer + binary cell enables
  DPA RNM         Verilog-AMS wreal real-number PA model (needs an AMS
                  simulator; its baked LUTs are self-checked vs the DPA)

The chain the files realize:
  env -> CFR clip -> [polar DPD LUT] -> DPA thermo decoder -> RNM DPA
  phase-command -> DTC phase acc -> EFM1 dither -> DTC code
"""
import os
import tempfile

from polartx.cal.polar_dpd import PolarDPD
from polartx.dpa import DPA, DPAConfig
from polartx.export import rtl

OUT = os.path.join(os.path.dirname(__file__), "out", "ex17_rtl")
os.makedirs(OUT, exist_ok=True)


def main():
    dpa = DPA(DPAConfig(n_bits=10, n_thermo=7, sigma_cell=0.01, gradient=0.02,
                        amam=("rapp", 2.5, 1.1), ampm_deg_poly=(0.0, 2.0, 3.0)))
    dpd = PolarDPD.from_dpa(dpa)

    paths = rtl.emit_datapath(dpa, OUT, dpd=dpd)
    print(f"emitted {len(paths)} files to {OUT}\n")

    print("bit-true verification (iverilog):")
    checks = [("CFR clip", rtl.verify_cfr_clip),
              ("DTC phase acc", rtl.verify_phase_acc),
              ("DPA thermo decoder", rtl.verify_thermo_decoder),
              ("polar DPD LUT", rtl.verify_with_iverilog)]
    for name, fn in checks:
        out = fn(OUT)
        if out is None:
            print(f"  {name:22s}: SKIP (iverilog not installed)")
        else:
            res = next((l for l in out.splitlines()
                        if "PASS" in l or "FAIL" in l), "?")
            print(f"  {name:22s}: {res}")

    sc = rtl.dpa_rnm_selfcheck(dpa, OUT)
    print(f"\nDPA RNM self-check (baked AMS LUT vs behavioral DPA):")
    print(f"  amp max err {sc['amp_max_err']:.2e}, "
          f"AM-PM max err {sc['ph_max_err']:.2e} rad -> "
          f"{'OK' if sc['ok'] else 'FAIL'}")
    print("\n  co-simulate dpa_rnm.vams + tb_dpa_ams.vams under a Verilog-AMS")
    print("  simulator (spectre/xcelium/questa-AMS) to drive the analog PA.")


if __name__ == "__main__":
    main()
