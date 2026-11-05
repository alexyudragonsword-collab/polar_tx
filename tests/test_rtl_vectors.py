"""Integer golden-vector sets for an external RTL lockstep (export/rtl_vectors.py):
the streams equal the bit-true references export/rtl.py already checks with its own
testbenches, and an export is reproducible byte for byte."""
import filecmp
import json

import numpy as np
import pytest

from polartx.export.rtl import (efm1_int_golden, emit_dpd_rtl, emit_dtc_dither_rtl,
                                phase_acc_golden, quantize_dpd_luts)
from polartx.export.rtl_vectors import DPA_KW, write_set


def _ints(p):
    return np.loadtxt(p, dtype=np.int64)


@pytest.mark.parametrize("name", ["dtc_random", "dtc_ramp"])
def test_dtc_sets_are_the_efm1_golden(tmp_path, name):
    meta = write_set(name, str(tmp_path))
    x, code = _ints(tmp_path / "phase_w.txt"), _ints(tmp_path / "code.txt")
    assert x.size == code.size == meta["n"]
    assert np.array_equal(code, efm1_int_golden(x, meta["n_bits"], meta["frac_bits"]))
    w = meta["n_bits"] + meta["frac_bits"]
    assert x.min() >= 0 and x.max() < (1 << w)
    if name == "dtc_ramp":
        # a phase accumulator stream: constant increment modulo 2^w, wrapping
        assert np.array_equal(x, phase_acc_golden(np.full(x.size, x[0]), w))
        assert np.sum(np.diff(x) < 0) > 10
    # error feedback makes the average code exact: mean(code) -> mean(x) / 2^frac
    # (modulo wrap aside), the property the dither exists for
    unwrapped = x / (1 << meta["frac_bits"])
    lsb = np.mean(code - unwrapped)
    assert abs(lsb) < 0.01


def test_dtc_random_matches_the_emitted_testbench_golden(tmp_path):
    write_set("dtc_random", str(tmp_path / "v"))
    emit_dtc_dither_rtl(str(tmp_path / "e"))
    rows = np.loadtxt(tmp_path / "e" / "golden_dither.csv", delimiter=",", dtype=np.int64)
    assert np.array_equal(rows[:, 0], _ints(tmp_path / "v" / "phase_w.txt"))
    assert np.array_equal(rows[:, 1], _ints(tmp_path / "v" / "code.txt"))


def test_dpd_set_is_the_quantized_lut(tmp_path):
    from polartx.cal.polar_dpd import PolarDPD
    from polartx.dpa import DPA, DPAConfig

    meta = write_set("dpd_lut", str(tmp_path / "v"))
    luts = quantize_dpd_luts(PolarDPD.from_dpa(DPA(DPAConfig(**DPA_KW))))
    emit_dpd_rtl(luts, str(tmp_path / "e"))
    for rom in ("dpd_amp.memh", "dpd_ph.memh"):     # the ROM images emit_dpd_rtl writes
        assert filecmp.cmp(tmp_path / "v" / rom, tmp_path / "e" / rom, shallow=False)
    addr = _ints(tmp_path / "v" / "addr.txt")
    n = 1 << meta["addr_bits"]
    assert np.array_equal(addr[:n], np.arange(n)) and addr.size == meta["n"] > n
    assert np.array_equal(_ints(tmp_path / "v" / "amp.txt"), luts["amp_code"][addr])
    assert np.array_equal(_ints(tmp_path / "v" / "ph.txt"), luts["ph_code"][addr])
    # the float LUT in the meta is what the codes quantize (within 1/2 LSB + rounding)
    amp_f, ph_f = np.array(meta["amp_float"]), np.array(meta["ph_float_rad"])
    assert np.max(np.abs(luts["amp_code"] / ((1 << meta["amp_bits"]) - 1) - amp_f)) \
        <= 0.5 / ((1 << meta["amp_bits"]) - 1) + 1e-12
    half = 1 << (meta["ph_bits"] - 1)
    assert np.max(np.abs(luts["ph_code"] * np.pi / half - ph_f)) <= 0.5 * np.pi / half + 1e-12


@pytest.mark.parametrize("name", ["dtc_ramp", "dpd_lut"])
def test_export_is_reproducible(tmp_path, name):
    write_set(name, str(tmp_path / "a"))
    write_set(name, str(tmp_path / "b"))
    cmp = filecmp.dircmp(tmp_path / "a", tmp_path / "b")
    assert not cmp.diff_files and not cmp.left_only and not cmp.right_only
    json.loads((tmp_path / "a" / sorted(p.name for p in (tmp_path / "a").glob("*_meta.json"))[0]).read_text())


def test_unknown_set_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_set("nope", str(tmp_path))
