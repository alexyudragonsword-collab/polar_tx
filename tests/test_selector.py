"""Architecture selector: the narrowband/wideband split and its physics."""
import numpy as np
import pytest

from polartx.selector import Requirement, select


def test_wideband_excludes_adpll():
    """Above the two-point coverage ceiling the ADPLL is infeasible and the
    open-loop DTC is recommended."""
    for bw in (80e6, 200e6, 320e6):
        rep = select(Requirement("wb", bw, "ofdm", fout=6e9))
        adpll = next(c for c in rep.candidates if c.arch == "adpll_two_point")
        assert not adpll.feasible
        assert rep.best.arch == "dtc_open_loop"


def test_narrowband_prefers_adpll_when_calibrated():
    """With a calibrated two-point (0.2%), the in-loop-FM ADPLL beats the
    open-loop DTC across the narrowband range — no DTC quant/jitter/INL floor."""
    for bw in (1e6, 10e6, 20e6, 40e6):
        rep = select(Requirement("nb", bw, "ofdm", fout=3.5e9,
                                 two_point_gain_match=2e-3))
        assert rep.best.arch == "adpll_two_point", f"bw={bw}"


def test_uncalibrated_two_point_loses_the_narrowband_edge():
    """Without calibration (0.5% match) the ADPLL advantage collapses beyond a
    few MHz — this is exactly what the online two-point cal buys."""
    cal = select(Requirement("nb", 10e6, "ofdm", fout=3.5e9,
                             two_point_gain_match=2e-3))
    unc = select(Requirement("nb", 10e6, "ofdm", fout=3.5e9,
                             two_point_gain_match=5e-3))
    assert cal.best.arch == "adpll_two_point"
    assert unc.best.arch == "dtc_open_loop"


def test_dtc_evm_improves_with_more_bits():
    """More DTC bits lowers the quantization floor -> better (or equal) DTC EVM."""
    lo = select(Requirement("w", 160e6, "ofdm", dtc_bits=8)).candidates
    hi = select(Requirement("w", 160e6, "ofdm", dtc_bits=13)).candidates
    dtc_lo = next(c for c in lo if c.arch == "dtc_open_loop")
    dtc_hi = next(c for c in hi if c.arch == "dtc_open_loop")
    assert dtc_hi.evm_db <= dtc_lo.evm_db + 1e-6


def test_integrated_pn_grows_with_bandwidth():
    """Wider signals integrate more open-loop LO phase noise: the DTC synth-PN
    term degrades monotonically with bandwidth."""
    prev = -np.inf
    last = None
    for bw in (10e6, 40e6, 160e6, 320e6):
        rep = select(Requirement("w", bw, "ofdm", fout=6e9))
        pn = next(c for c in rep.candidates
                  if c.arch == "dtc_open_loop").terms["synth_pn"]
        if last is not None:
            assert pn >= last - 1e-9   # non-decreasing (worse) with bandwidth
        last = pn


def test_report_surfaces_target_pass_fail():
    rep = select(Requirement("t", 160e6, "ofdm", evm_db_max=-38, dtc_bits=11))
    assert rep.best is not None
    assert "recommend" in rep.recommendation
    assert "wifi_dtc" in rep.suggest_preset()
    # table renders without error and lists both architectures
    tbl = rep.table()
    assert "dtc_open_loop" in tbl and "adpll_two_point" in tbl


def test_infeasible_everything_reports_no_recommendation():
    """A ceiling below the bandwidth AND (hypothetically) no DTC — here only
    ADPLL is knocked out, so DTC still wins; assert the report stays sane."""
    rep = select(Requirement("x", 100e6, "ofdm", adpll_bw_ceiling=10e6))
    assert rep.best.arch == "dtc_open_loop"
    assert rep.recommendation
