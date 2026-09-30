"""Cartesian (digital I/Q) sibling chain — the stage-2 acceptance list.

Same WiFi 160 MHz 1024-QAM burst, same CFR, same seed as the polar chain it
is compared with.  The five acceptance items from the design note are
tests 1-5; the rest pin the preset and the ChainConfig contract.
"""
import warnings

import numpy as np
import pytest

from polartx import (DPA, CartesianTX, ChainConfig, DPAConfig,
                     IdealPhaseModulator, PolarTX, RFDAC, RFDACConfig,
                     wifi_dtc, wifi_rfdac)
from polartx.cartesian import IGNORED_CHAIN_FIELDS, ignored_chain_settings
from polartx.dpa.dpa import efficiency_curve

CFR_DB = 8.5


@pytest.fixture(scope="module")
def ideal_polar():
    """Ideal polar chain (as in test_outphasing): passthrough phase
    modulator, 14-bit linear DPA, no mismatch, no envelope floor."""
    dpa = DPA(DPAConfig(n_bits=14, sigma_cell=0.0, amam="ideal"))
    return PolarTX(ChainConfig(cfr_papr_db=CFR_DB, env_floor=0.0),
                   IdealPhaseModulator(), dpa), dpa


@pytest.fixture(scope="module")
def wf():
    return wifi_dtc(bw=160e6, qam=1024).make_waveform(n_symbols=4, seed=0)


def _ideal_cartesian(n_bits=14, **rf):
    return CartesianTX(ChainConfig(cfr_papr_db=CFR_DB),
                       RFDAC(RFDACConfig(n_bits=n_bits, sigma_cell=0.0, **rf)))


# ------------------------------ 1. ideal chain equals ideal polar
def test_ideal_cartesian_matches_ideal_polar_within_0p1_db(ideal_polar, wf):
    """Acceptance 1.  With no impairment anywhere, both topologies must hand
    the metric layer the same CFR-limited signal: the only thing left in
    either EVM is the clipping residual of the shared waveform."""
    polar, _ = ideal_polar
    e_polar = polar.run(wf, noise=False, seed=1).evm().db
    e_cart = _ideal_cartesian().run(wf, noise=False, seed=1).evm().db
    assert abs(e_polar - e_cart) <= 0.1, (e_polar, e_cart)    # measured 0.001
    assert e_polar < -35.0                # a CFR floor, not a broken chain


def test_resolution_matters_below_14_bits_and_not_above(wf):
    """The 14-bit floor is genuinely the CFR's: 6 bits is visibly worse
    (the 4x oversampling spreads 3/4 of the quantization noise out of band,
    so even 8 bits only costs 0.3 dB here), 16 bits is not visibly better."""
    e = {n: _ideal_cartesian(n).run(wf, noise=False).evm().db for n in (6, 14, 16)}
    assert e[6] > e[14] + 2.0                            # measured -38.6 vs -41.6
    assert abs(e[16] - e[14]) < 0.1


# ---------------------------------------- 2. I/Q imbalance -> image
def test_iq_imbalance_costs_the_evm_the_image_rejection_says(wf):
    """Acceptance 2 on the modulated chain.  An OFDM image lands on the
    mirror subcarrier as an in-band error at -IRR relative to the signal,
    so the EVM degradation (floor subtracted in power) must track IRR."""
    from polartx import iq_image_rejection_db
    e0 = _ideal_cartesian().run(wf, noise=False).evm().db
    for g_db, ph in ((0.1, 1.0), (0.3, 2.0)):
        e = _ideal_cartesian(iq_gain_db=g_db, iq_phase_deg=ph).run(
            wf, noise=False).evm().db
        degradation = 10 * np.log10(10 ** (e / 10) - 10 ** (e0 / 10))
        assert abs(degradation + iq_image_rejection_db(g_db, ph)) < 1.0, (
            g_db, ph, degradation)      # measured -39.4 vs IRR 39.6 at 0.1/1


# ---------------------------------- 3. same mismatch model as polar
def test_cell_mismatch_is_the_dpa_model_and_its_error_scales_as_sigma_squared(wf):
    """Acceptance 3 on the chain.  The mismatch tables are pinned equal to
    the DPA's in test_rfdac; here the chain must actually consume them.
    The error the mismatch adds to the output is linear in the unit
    deviation, so its power must grow 20 dB per decade of sigma_cell —
    and on a 6+4 segmented 10-bit array at 1 % it sits ~30 dB under the
    CFR floor, which is why it does NOT move the EVM (a real property of
    segmentation, not a knob nobody wired: the error itself is asserted)."""
    kw = dict(n_bits=10, n_thermo=6, seed=3)

    def run(sigma):
        return CartesianTX(ChainConfig(cfr_papr_db=CFR_DB),
                           RFDAC(RFDACConfig(sigma_cell=sigma, **kw))).run(wf, noise=False)

    r0 = run(0.0)
    p_sig = np.mean(np.abs(r0.y) ** 2)
    err_db = {s: 10 * np.log10(np.mean(np.abs(run(s).y - r0.y) ** 2) / p_sig)
              for s in (0.01, 0.1)}
    assert -80.0 < err_db[0.01] < -60.0                  # measured -71.9 dB
    assert err_db[0.1] - err_db[0.01] == pytest.approx(20.0, abs=1.0)   # 20.0
    assert abs(run(0.01).evm().db - r0.evm().db) < 0.1   # 30 dB under the floor
    dpa = DPA(DPAConfig(sigma_cell=0.01, **kw))
    assert run(0.01).rfdac.mismatch()["i"]["inl_max"] == dpa.inl_dnl()["inl_max"]
    assert np.array_equal(run(0.01).y, run(0.01).y)     # deterministic


# ------------------------------------------ 4. efficiency vs backoff
def test_rfdac_is_below_the_polar_dpa_at_6_db_backoff_and_on_the_burst(wf):
    """Acceptance 4 (direction only).  At a 6 dB CW backoff the SCPA law
    keeps more than the RF-DAC's |I|+|Q| law on its best axis, let alone
    on the diagonal; and on the modulated burst the RF-DAC's average sits
    below the polar DPA's with the same eta_peak."""
    eta_peak = 0.85
    p = wifi_dtc(bw=160e6, qam=1024)
    scpa = p.tx.dpa.cfg.eff
    assert scpa[0] == "scpa" and scpa[2] == eta_peak
    rf = RFDAC(RFDACConfig(n_bits=12, eff=("iq_cells", eta_peak)))
    x = 10 ** (-6.0 / 20)                                # 6 dB below full scale
    eta_scpa = float(efficiency_curve(scpa, np.array([x]))[0])
    assert rf.efficiency(x, 0.0) < eta_scpa               # 42.5 % vs 50.9 %
    assert rf.efficiency(x / np.sqrt(2), x / np.sqrt(2)) < rf.efficiency(x, 0.0)

    r_polar = p.tx.run(wf, noise=False, seed=1)
    r_cart = CartesianTX(ChainConfig(cfr_papr_db=p.tx.cfg.cfr_papr_db), rf).run(
        wf, noise=False, seed=1)
    ep, ec = r_polar.avg_efficiency(p.tx.dpa), r_cart.avg_efficiency()
    assert ec["eta_avg"] < ep["eta_avg"]                  # measured 28.3 % vs 42.8 %
    assert set(ec) == set(ep)
    assert r_cart.info["backoff_db"] == ec["backoff_db"] > 6.0


# ------------------------------------ 5. no envelope path: warn, ignore
def test_env_skew_is_warned_about_and_has_no_effect(ideal_polar, wf):
    """Acceptance 5.  A ChainConfig knob that names a path this chain does
    not have must be reported, not silently absorbed — and the output must
    be identical with or without it.  The polar contrast is asserted too,
    so the invariance is not the vacuous kind."""
    _, dpa = ideal_polar
    rf = RFDAC(RFDACConfig(n_bits=14))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clean = CartesianTX(ChainConfig(cfr_papr_db=CFR_DB), rf)   # no warning
    with pytest.warns(UserWarning, match="env_skew_s"):
        skewed = CartesianTX(ChainConfig(cfr_papr_db=CFR_DB, env_skew_s=0.5e-9), rf)
    assert skewed.ignored == ["env_skew_s"]
    r0, r1 = clean.run(wf, noise=False, seed=1), skewed.run(wf, noise=False, seed=1)
    assert np.array_equal(r0.y, r1.y)
    assert r1.info["ignored"] == ["env_skew_s"] and r0.info["ignored"] == []

    polar = PolarTX(ChainConfig(cfr_papr_db=CFR_DB, env_floor=0.0),
                    IdealPhaseModulator(), dpa)
    polar_skewed = PolarTX(ChainConfig(cfr_papr_db=CFR_DB, env_floor=0.0,
                                       env_skew_s=0.5e-9), IdealPhaseModulator(), dpa)
    assert polar_skewed.run(wf, noise=False).evm().db > polar.run(wf, noise=False).evm().db + 10


def test_every_envelope_and_phase_field_is_warned_about():
    """The ignore list must cover every ChainConfig field that is not
    cfr_papr_db, and each one must trigger the warning when moved off its
    default — a new ChainConfig field cannot be silently absorbed."""
    from dataclasses import fields
    names = {f.name for f in fields(ChainConfig)}
    assert names == set(IGNORED_CHAIN_FIELDS) | {"cfr_papr_db"}
    assert ignored_chain_settings(ChainConfig()) == []
    assert ignored_chain_settings(ChainConfig(cfr_papr_db=6.0)) == []
    assert ignored_chain_settings(ChainConfig(env_floor=0.1, interleave=2)) == [
        "env_floor", "interleave"]


# ------------------------------------------------- result contract
def test_result_answers_the_same_metric_set_as_polar(wf):
    r = _ideal_cartesian().run(wf, noise=True, seed=1)
    assert r.evm_equalize_default == "scalar"
    assert r.evm(equalize="scalar").db == r.evm().db
    a = r.aclr()
    assert a["upper_dbc"] < -40.0 and a["lower_dbc"] < -40.0
    f, pdb = r.psd(nfft=2048)
    assert len(f) == len(pdb) == 2048
    ok, margin, _ = r.check_mask()                       # PolarResult's tuple
    assert isinstance(bool(ok), bool) and np.isfinite(margin)
    assert r.i_code.dtype == np.int64 and r.i_code.shape == wf.x.shape
    assert abs(r.i_code).max() == r.rfdac.cfg.full_scale_code or \
        abs(r.q_code).max() == r.rfdac.cfg.full_scale_code   # one axis peaks at FS
    assert r.info["fs_scale"] > 0 and r.info["cfr_papr_db"] == CFR_DB


# ---------------------------------------------- preset + registry
def test_preset_is_registered_as_a_standard_chain_and_reports():
    from polartx.guiutil import PRESETS, build_preset, run_chain_report
    name = "WiFi 160 MHz (RF-DAC)"
    assert name in PRESETS and not name.startswith("Bench:")
    p = build_preset(name, n_bits=9)                     # GUI-style override
    assert isinstance(p.tx, CartesianTX)
    assert p.tx.rfdac.cfg.n_bits == 9                    # n_bits is the DAC's
    assert p.tx.cfg.cfr_papr_db == p.polar_tx.cfg.cfr_papr_db
    rep = run_chain_report(name, seed=1, n_units=3)
    assert rep["metrics"].get("mask") in ("PASS", "FAIL")
    assert "EVM [dB]" in rep["metrics"]
    import matplotlib.pyplot as plt
    plt.close(rep["fig"])


def test_preset_shares_the_frequency_plan_with_wifi_dtc():
    c = wifi_rfdac(bw=160e6, qam=1024)
    d = wifi_dtc(bw=160e6, qam=1024)
    assert c.fs_bb == d.fs_bb
    assert c.polar_tx.phasemod.cfg == d.tx.phasemod.cfg
    assert c.tx.rfdac.cfg.fout == d.tx.phasemod.cfg.fout
    assert c.tx.rfdac.cfg.sigma_cell == d.tx.dpa.cfg.sigma_cell
    r = c.tx.run(c.make_waveform(n_symbols=3, seed=0), noise=True, seed=1)
    assert r.evm_equalize_default == "scalar"
    assert r.evm().db < -30.0                            # measured -40.6 dB
    with pytest.warns(UserWarning, match="env_skew_s"):
        wifi_rfdac(bw=160e6, qam=1024, env_skew_s=1e-9)  # forwarded, then warned
