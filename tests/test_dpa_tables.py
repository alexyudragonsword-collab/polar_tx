"""DPA code tables: mismatch statistics and AM-AM/AM-PM laws."""
import numpy as np

from polartx.dpa import DPA, DPAConfig, code_amplitude_table
from polartx.dpa.characteristics import amam_curve


def test_encode_exact_grid():
    dpa = DPA(DPAConfig(n_bits=8))
    codes = np.arange(256)
    assert (dpa.encode(codes / 255.0) == codes).all()


def test_mismatch_sqrtN_law():
    """Random unit mismatch accumulates as sqrt(enabled units): mid-code
    amplitude sigma over seeds ~ sigma_cell * sqrt(N/2)."""
    n_bits, sigma = 10, 0.02
    mid = 1 << (n_bits - 1)
    amps = [code_amplitude_table(n_bits, n_bits, sigma, 0.0,
                                 np.random.default_rng(s))[mid]
            for s in range(400)]
    expect = sigma * np.sqrt(mid)
    assert abs(np.std(amps) - expect) / expect < 0.15


def test_thermo_vs_binary_dnl():
    """Binary-heavy segmentation shows the classic MSB-transition DNL
    step; fully thermometer stays sub-LSB at the same mismatch."""
    rng = lambda: np.random.default_rng(7)
    from polartx.dpa import inl_dnl
    d_bin = inl_dnl(code_amplitude_table(10, 0, 0.05, 0.0, rng()))
    d_th = inl_dnl(code_amplitude_table(10, 10, 0.05, 0.0, rng()))
    assert d_bin["dnl_max"] > 3.0 * d_th["dnl_max"]


def test_ampm_poly_in_tables():
    poly = (0.0, 2.0, 3.0)          # deg: 2*r + 3*r^2
    dpa = DPA(DPAConfig(n_bits=8, ampm_deg_poly=poly))
    r = np.arange(256) / 255.0
    expect = np.deg2rad(2.0 * r + 3.0 * r ** 2)
    assert np.allclose(dpa.phase_table, expect, atol=1e-12)


def test_rapp_compression():
    r = np.linspace(0.01, 1.0, 100)
    y = amam_curve(("rapp", 2.0, 1.5), r)
    gain = y / r
    assert y[-1] == 1.0                       # normalized full scale
    assert gain[0] > 1.05 * gain[-1]          # small-signal gain > FS gain
    assert (np.diff(y) > 0).all()             # monotone
