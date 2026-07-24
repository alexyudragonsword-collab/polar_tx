"""RTL export of the polar-DPD dual LUT: quantization + iverilog golden."""
import numpy as np
import pytest

from polartx.cal.polar_dpd import PolarDPD
from polartx.dpa import DPA, DPAConfig
from polartx.export.rtl import (emit_dpd_rtl, quantize_dpd_luts,
                                verify_with_iverilog)

DPA_CFG = DPAConfig(n_bits=10, amam=("rapp", 2.5, 1.1),
                    ampm_deg_poly=(0.0, 2.0, 3.0))


def _luts():
    return quantize_dpd_luts(PolarDPD.from_dpa(DPA(DPA_CFG)))


def test_quantization_accuracy():
    """Fixed-point LUTs stay within an LSB of the float LUTs."""
    dpd = PolarDPD.from_dpa(DPA(DPA_CFG))
    q = _luts()
    x = np.arange(1 << q["addr_bits"]) / ((1 << q["addr_bits"]) - 1)
    amp, ph = dpd.predistort(x)
    assert np.max(np.abs(q["amp_float"] - amp)) < 1.5 / (1 << q["amp_bits"])
    assert np.max(np.abs(q["ph_float"] - ph)) < 1.5 * np.pi / (1 << (q["ph_bits"] - 1))


def test_emit_files(tmp_path):
    paths = emit_dpd_rtl(_luts(), str(tmp_path))
    for name in ("polar_dpd_lut.v", "tb_polar_dpd_lut.v", "dpd_amp.memh",
                 "dpd_ph.memh", "golden_dpd.csv"):
        assert name in paths
    v = open(paths["polar_dpd_lut.v"]).read()
    assert "module polar_dpd_lut" in v and "$readmemh" in v


def test_iverilog_golden(tmp_path):
    paths = emit_dpd_rtl(_luts(), str(tmp_path))
    out = verify_with_iverilog(str(tmp_path))
    if out is None:
        pytest.skip("iverilog not installed")
    assert "PASS" in out and "FAIL" not in out


def test_dtc_dither_rtl_matches_python_engine(tmp_path):
    """The EFM1 dither Verilog is bit-exact against BOTH the integer
    golden and the float engine's diff-of-floor-of-cumsum identity."""
    from polartx.export.rtl import (efm1_int_golden, emit_dtc_dither_rtl,
                                    verify_dither_with_iverilog)
    from polartx.phasemod.dtc_openloop import _efm1_quantize
    rng = np.random.default_rng(1)
    xw = rng.integers(0, 1 << 19, 500)
    g = efm1_int_golden(xw, 11, 8)
    f = np.mod(_efm1_quantize(xw / 256.0), 1 << 11).astype(int)
    assert np.array_equal(g, f)

    emit_dtc_dither_rtl(str(tmp_path))
    out = verify_dither_with_iverilog(str(tmp_path))
    if out is None:
        pytest.skip("iverilog not installed")
    assert "PASS" in out and "FAIL" not in out
