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
