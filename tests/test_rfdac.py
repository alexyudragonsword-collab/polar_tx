"""RF-DAC output stage: the laws the Cartesian chain's acceptance rests on.

Every assertion is a property of the model (a closed form, an identity with
the polar DPA, a gate) — never a snapshot of an output.
"""
import numpy as np
import pytest

from polartx import DPA, DPAConfig, RFDAC, RFDACConfig, iq_image_rejection_db
from polartx.dpa.mismatch import inl_dnl


def _tone(n=4096, k=37, amp=0.9):
    return amp * np.exp(1j * 2 * np.pi * k * np.arange(n) / n)


def _sig_img_db(y, k):
    """Signal-to-image power ratio [dB] of a single-tone output at bin k."""
    spec = np.fft.fft(y) / len(y)
    img = max(abs(spec[-k]) ** 2, 1e-30)              # an ideal DAC has none
    return float(10 * np.log10(abs(spec[k]) ** 2 / img))


# ------------------------------------------------------------- coding
def test_sign_magnitude_coding_spans_both_polarities_and_clips():
    d = RFDAC(RFDACConfig(n_bits=8))
    fs = d.cfg.full_scale_code
    assert fs == 255
    code = d.encode(np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.7]))
    assert code.tolist() == [-255, -128, 0, 128, 255, 255]     # rint, then clip
    y = d(code, np.zeros_like(code), noise=False)
    assert np.allclose(y.real, [-1.0, -128 / 255, 0.0, 128 / 255, 1.0, 1.0])
    assert np.all(y.imag == 0.0)


def test_ideal_dac_is_the_identity_to_within_half_an_lsb():
    d = RFDAC(RFDACConfig(n_bits=12))
    x = _tone()
    y = d(d.encode(x.real), d.encode(x.imag), noise=False)
    assert np.abs(y - x).max() <= 0.5 * np.sqrt(2) / d.cfg.full_scale_code


# ------------------------------------------------ I/Q imbalance -> image
def test_image_rejection_matches_the_closed_form_and_the_textbook_small_angle():
    """Acceptance 2.  A CW tone through the imbalanced DAC: the measured
    signal-to-image ratio must land within 1 dB of the analytic IRR, which
    itself must agree with the textbook 4/(g^2 + phi^2) at 0.1 dB / 1 deg."""
    g_db, ph = 0.1, 1.0
    d = RFDAC(RFDACConfig(n_bits=14, iq_gain_db=g_db, iq_phase_deg=ph))
    x = _tone()
    meas = _sig_img_db(d(d.encode(x.real), d.encode(x.imag), noise=False), 37)
    irr = iq_image_rejection_db(g_db, ph)
    assert abs(meas - irr) < 1.0, (meas, irr)          # measured 39.61 vs 39.61
    g = 10 ** (g_db / 20) - 1
    textbook = 10 * np.log10(4 / (g ** 2 + np.deg2rad(ph) ** 2))
    assert abs(irr - textbook) < 0.1                    # 39.61 vs 39.60
    assert 35.0 < irr < 45.0                            # the familiar ~40 dB class


def test_image_grows_with_either_error_and_vanishes_without():
    assert iq_image_rejection_db(0.0, 0.0) > 200.0      # no image at all
    assert iq_image_rejection_db(0.2, 0.0) < iq_image_rejection_db(0.1, 0.0)
    assert iq_image_rejection_db(0.0, 2.0) < iq_image_rejection_db(0.0, 1.0)
    d = RFDAC(RFDACConfig(n_bits=14))
    x = _tone()
    y = d(d.encode(x.real), d.encode(x.imag), noise=False)
    assert _sig_img_db(y, 37) > 70.0                    # quantization only


# --------------------------------------------------------- LO leakage
def test_lo_leakage_is_a_dc_term_at_the_stated_level():
    d = RFDAC(RFDACConfig(n_bits=14, lo_leakage_dbc=-30.0))
    x = _tone(amp=1.0)
    y = d(d.encode(x.real), d.encode(x.imag), noise=False)
    dc = abs(np.mean(y))
    assert 20 * np.log10(dc) == pytest.approx(-30.0, abs=0.05)
    d0 = RFDAC(RFDACConfig(n_bits=14))
    y0 = d0(d0.encode(x.real), d0.encode(x.imag), noise=False)
    assert abs(np.mean(y0)) < 1e-3                      # tone has no DC


# ------------------------------------------------------- clock jitter
def test_clock_jitter_is_white_phase_scaled_by_fout_and_gated_by_noise():
    """The DTC convention: sigma_phi = 2 pi fout sigma_tau per sample."""
    tau, fout = 50e-15, 5.9e9
    d = RFDAC(RFDACConfig(n_bits=14, jitter_rms_s=tau, fout=fout))
    n = 200_000
    code = np.full(n, d.cfg.full_scale_code)
    y = d(code, np.zeros(n, dtype=np.int64), noise=True, seed=3)
    phi = np.angle(y)
    assert np.std(phi) == pytest.approx(2 * np.pi * fout * tau, rel=0.02)
    y_off = d(code, np.zeros(n, dtype=np.int64), noise=False)
    assert np.all(y_off == 1.0)                         # gated off entirely
    d2 = RFDAC(RFDACConfig(n_bits=14, jitter_rms_s=tau, fout=2 * fout))
    phi2 = np.angle(d2(code, np.zeros(n, dtype=np.int64), noise=True, seed=3))
    assert np.std(phi2) == pytest.approx(2 * np.std(phi), rel=0.02)


def test_lo_phase_noise_uses_the_dtc_generator_and_needs_fs():
    """The RF-DAC's carrier has the same Leeson phase noise the DTC's LO
    has: same OscConfig, same locked-LO flattening, gated by noise.  With
    the WiFi plan's LO it sets the RF-DAC chain's floor at the polar
    chain's level (test_selector pins the two floors equal)."""
    from polartx.vendor.pllsim.blocks.oscillator import OscConfig
    lo = OscConfig(f0=5.9e9, gain=1.0, pn_dbchz=-115.0, pn_foffset=1e6,
                   pn_f1f3=2e5, pn_floor_dbchz=-155.0)
    d = RFDAC(RFDACConfig(n_bits=14, lo_pn=lo, lo_loop_bw=400e3))
    n, fs = 65536, 640e6
    code = np.full(n, d.cfg.full_scale_code)
    zero = np.zeros(n, dtype=np.int64)
    with pytest.raises(ValueError):
        d(code, zero, noise=True)                         # fs is required
    y = d(code, zero, noise=True, seed=2, fs=fs)
    phi = np.unwrap(np.angle(y))
    # the phase noise integrated over the 640 MHz Nyquist band, LO-limited:
    # -115 dBc/Hz at 1 MHz falling as 1/f^2 to the -155 floor -> a few mrad
    assert 1e-3 < np.std(phi) < 3e-2
    assert np.all(d(code, zero, noise=False) == 1.0)      # gated off
    d0 = RFDAC(RFDACConfig(n_bits=14))
    assert np.all(d0(code, zero, noise=True, seed=2) == 1.0)   # no LO given


# -------------------------------------- same mismatch model as the DPA
def test_cell_mismatch_reproduces_the_polar_dpa_inl_dnl_exactly():
    """Acceptance 3.  Built from a DPA's (n_bits, n_thermo, sigma_cell,
    gradient, seed), the I array IS that DPA's array: identical INL/DNL
    vectors, not merely similar statistics.  Q draws with seed + 1 so the
    two arrays are independent, as two physical arrays are."""
    kw = dict(n_bits=10, n_thermo=6, sigma_cell=0.002, gradient=0.001, seed=3)
    dpa = DPA(DPAConfig(**kw))
    rf = RFDAC(RFDACConfig(**kw))
    m = rf.mismatch()
    ref = dpa.inl_dnl()
    for key in ("inl_lsb", "dnl_lsb"):
        assert np.array_equal(m["i"][key], ref[key])
    assert m["i"]["inl_max"] == ref["inl_max"] > 0.0    # measured 0.045 LSB
    assert not np.array_equal(m["q"]["inl_lsb"], m["i"]["inl_lsb"])
    q_as_i = RFDAC(RFDACConfig(**{**kw, "seed": kw["seed"] + 1})).mismatch()["i"]
    assert np.array_equal(q_as_i["inl_lsb"], m["q"]["inl_lsb"])
    assert np.array_equal(inl_dnl(rf._raw_i)["dnl_lsb"], m["i"]["dnl_lsb"])


def test_no_mismatch_means_perfectly_linear_tables():
    rf = RFDAC(RFDACConfig(n_bits=10, sigma_cell=0.0))
    fs = rf.cfg.full_scale_code
    assert np.allclose(rf.amp_table_i, np.arange(fs + 1) / fs)
    assert rf.mismatch()["i"]["inl_max"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------- efficiency
def test_efficiency_law_pays_for_abs_i_plus_abs_q():
    """eta_peak on an axis at full scale, eta_peak/sqrt(2) on the diagonal
    at the same output power, and linear in the drive on an axis."""
    eta = 0.85
    rf = RFDAC(RFDACConfig(eff=("iq_cells", eta)))
    assert rf.efficiency(1.0, 0.0) == pytest.approx(eta)
    assert rf.efficiency(0.0, -1.0) == pytest.approx(eta)
    r = 1 / np.sqrt(2)
    assert rf.efficiency(r, r) == pytest.approx(eta / np.sqrt(2))
    assert rf.efficiency(0.5, 0.0) == pytest.approx(0.5 * eta)
    assert rf.efficiency(0.0, 0.0) == 0.0
    with pytest.raises(ValueError):
        RFDAC(RFDACConfig(eff=("scpa", 0.67, 0.85)))


def test_average_efficiency_is_the_power_weighted_mean():
    rf = RFDAC(RFDACConfig(n_bits=10))
    fs = rf.cfg.full_scale_code
    i = np.array([fs, 0, fs // 2, -fs // 2])
    q = np.array([0, fs, fs // 2, fs // 2])
    a = rf.average_efficiency(i, q)
    ii, qq = i / fs, q / fs
    p_out = ii ** 2 + qq ** 2
    p_dc = (abs(ii) + abs(qq)) / rf.cfg.eff[1]
    assert a["eta_avg"] == pytest.approx(p_out.sum() / p_dc.sum())
    assert a["p_out_norm"] == pytest.approx(p_out.mean())
    assert a["backoff_db"] == pytest.approx(-10 * np.log10(p_out.mean()))
    assert set(a) == {"eta_avg", "p_out_norm", "backoff_db"}   # DPA's keys
