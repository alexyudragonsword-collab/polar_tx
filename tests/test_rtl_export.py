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


# --------------------------------------------------------- item-4 datapath

def test_thermo_decode_golden_big_thermometer():
    """The thermometer word is a big-int (2^127-1 for 127 segments) — the
    Python golden must not overflow."""
    from polartx.export.rtl import thermo_decode_golden
    g = thermo_decode_golden(10, 7)          # 127-segment thermometer
    assert g["n_seg"] == 127
    assert g["thermo_int"][-1] == (1 << 127) - 1   # top code: all segments on
    assert g["thermo_int"][8] == 1                 # code 8 = 1 segment (n_bin=3)
    assert int(g["bin"][11]) == 3


@pytest.mark.parametrize("n_bits,n_thermo", [(10, 7), (10, 0), (8, 8), (12, 9)])
def test_thermo_decoder_iverilog(tmp_path, n_bits, n_thermo):
    from polartx.dpa import DPAConfig
    from polartx.export.rtl import (emit_thermo_decoder_rtl,
                                    verify_thermo_decoder)
    emit_thermo_decoder_rtl(DPAConfig(n_bits=n_bits, n_thermo=n_thermo),
                            str(tmp_path))
    out = verify_thermo_decoder(str(tmp_path))
    if out is None:
        pytest.skip("iverilog not installed")
    assert "PASS" in out and "FAIL" not in out


def test_cfr_clip_golden_and_iverilog(tmp_path):
    from polartx.export.rtl import (cfr_clip_golden, emit_cfr_clip_rtl,
                                    verify_cfr_clip)
    e = np.array([0, 100, 3071, 3072, 3073, 4095])
    assert list(cfr_clip_golden(e, 3072)) == [0, 100, 3071, 3072, 3072, 3072]
    emit_cfr_clip_rtl(str(tmp_path), w=12, threshold=3072)
    out = verify_cfr_clip(str(tmp_path))
    if out is None:
        pytest.skip("iverilog not installed")
    assert "PASS" in out and "FAIL" not in out


def test_phase_acc_golden_and_iverilog(tmp_path):
    from polartx.export.rtl import (emit_phase_acc_rtl, phase_acc_golden,
                                    verify_phase_acc)
    # modulo-2^4 wrap: 10+10=20 -> 4, +10 -> 14, +10 -> 8
    assert list(phase_acc_golden([10, 10, 10, 10], 4)) == [10, 4, 14, 8]
    emit_phase_acc_rtl(str(tmp_path), w=16, n_vec=512)
    out = verify_phase_acc(str(tmp_path))
    if out is None:
        pytest.skip("iverilog not installed")
    assert "PASS" in out and "FAIL" not in out


def test_dpa_rnm_vams_selfcheck(tmp_path):
    """The baked Verilog-AMS real LUTs reproduce the DPA tables exactly."""
    from polartx.dpa import DPA, DPAConfig
    from polartx.export.rtl import dpa_rnm_selfcheck, emit_dpa_rnm_vams
    dpa = DPA(DPAConfig(n_bits=9, n_thermo=6, sigma_cell=0.02,
                        amam=("rapp", 2.0, 1.2), ampm_deg_poly=(1.0, 3.0)))
    emit_dpa_rnm_vams(dpa, str(tmp_path))
    sc = dpa_rnm_selfcheck(dpa, str(tmp_path))
    assert sc["ok"] and sc["amp_max_err"] < 1e-8 and sc["ph_max_err"] < 1e-8
    # the AMS files exist
    assert (tmp_path / "dpa_rnm.vams").exists()
    assert (tmp_path / "tb_dpa_ams.vams").exists()


def test_emit_datapath_full_chain(tmp_path):
    from polartx.dpa import DPA, DPAConfig
    from polartx.cal.polar_dpd import PolarDPD
    from polartx.export.rtl import emit_datapath
    dpa = DPA(DPAConfig(n_bits=10, n_thermo=7, amam=("rapp", 2.5, 1.1),
                        ampm_deg_poly=(0.0, 2.0)))
    paths = emit_datapath(dpa, str(tmp_path), dpd=PolarDPD.from_dpa(dpa))
    for f in ("cfr_clip.v", "dtc_phase_acc.v", "dpa_thermo_decode.v",
              "dpa_rnm.vams", "polar_dpd_lut.v"):
        assert f in paths and (tmp_path / f).exists()
