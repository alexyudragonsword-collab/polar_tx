"""Outphasing (LINC / Chireix) sibling chain — the stage-1 acceptance list.

Every assertion here is a physical property of the topology or of its
relationship to the polar chain it wraps, measured on the same WiFi 160 MHz
1024-QAM burst with the same seed.  The five acceptance items from the
design note map onto tests 2-6 below; the rest pin the combiner laws the
acceptance rests on.
"""
import numpy as np
import pytest

from polartx import (DPA, ChainConfig, DPAConfig, IdealPhaseModulator,
                     OutphasingCombiner, OutphasingTX, PolarTX, wifi_dtc,
                     wifi_outphasing)
from polartx.outphasing import (outphasing_decompose,
                                phase_mismatch_evm_budget_db, theta_stats)
from polartx.vendor.padpd.cfr import cfr_clip_filter

CFR_DB = 8.5


@pytest.fixture(scope="module")
def ideal():
    """Ideal polar chain: passthrough phase modulator, 14-bit linear DPA
    with no mismatch, no envelope hole-punching -- so the only thing left
    in its EVM is the CFR clipping residual of the waveform itself."""
    dpa = DPA(DPAConfig(n_bits=14, sigma_cell=0.0, amam="ideal"))
    cfg = ChainConfig(cfr_papr_db=CFR_DB, env_floor=0.0)
    return PolarTX(cfg, IdealPhaseModulator(), dpa), dpa


@pytest.fixture(scope="module")
def wf():
    return wifi_dtc(bw=160e6, qam=1024).make_waveform(n_symbols=4, seed=0)


def _lossless(mode="isolated", **kw):
    return OutphasingCombiner(mode=mode, combiner_loss_db=0.0, **kw)


# ------------------------------------------------------ 1. decomposition
def test_decomposition_reconstructs_the_signal_exactly(wf):
    x = wf.x
    a_max = float(np.abs(x).max())
    s1, s2, theta = outphasing_decompose(x, a_max)
    assert np.allclose(s1 + s2, x, atol=1e-12 * a_max)      # LINC identity
    assert np.allclose(np.abs(s1), a_max / 2) and np.allclose(np.abs(s2), a_max / 2)
    assert theta.min() >= 0.0 and theta.max() <= np.pi / 2 + 1e-12
    assert theta[np.argmax(np.abs(x))] == pytest.approx(0.0)  # peak -> in phase


# ------------------------------------ 2. ideal chain equals ideal polar
def test_ideal_outphasing_matches_ideal_polar_within_0p1_db(ideal, wf):
    """Acceptance 1.  With no impairment anywhere, both topologies must
    hand the metric layer the same CFR-limited signal.  A gap here means
    the decomposition or the combine is wrong, not that outphasing is
    'different' -- the math says s1 + s2 == x exactly."""
    polar, _ = ideal
    e_polar = polar.run(wf, noise=False, seed=1).evm().db
    e_outph = OutphasingTX(polar, _lossless()).run(wf, noise=False, seed=1).evm().db
    assert abs(e_polar - e_outph) <= 0.1, (e_polar, e_outph)   # measured 0.000
    assert e_polar < -35.0            # and the floor is the CFR one, not a bug


# --------------------------------- 3. branch phase mismatch vs the budget
def test_phase_mismatch_tracks_the_analytic_budget_and_is_monotonic(ideal, wf):
    """Acceptance 2.  A static phase error on branch 2 adds the known error
    vector (e^{j delta} - 1) s2; the closed-form budget scores that vector
    the way the OFDM EVM does (in-band, scalar-equalized).  The chain's
    measured degradation -- mismatch EVM with the CFR floor subtracted in
    power -- must land within 1 dB at 1 deg, and grow with the angle."""
    polar, _ = ideal
    x = cfr_clip_filter(wf.x, CFR_DB, wf.fs, wf.bw)
    _, s2, _ = outphasing_decompose(x, float(np.abs(x).max()))

    def evm_db(delta_deg):
        c = _lossless(phase_imbalance_deg=(0.0, delta_deg))
        return OutphasingTX(polar, c).run(wf, noise=False, seed=1).evm().db

    e0 = evm_db(0.0)
    for delta in (1.0, 2.0):
        e = evm_db(delta)
        degradation = 10 * np.log10(10 ** (e / 10) - 10 ** (e0 / 10))
        budget = phase_mismatch_evm_budget_db(x, s2, delta, wf.fs, wf.bw)
        assert abs(degradation - budget) < 1.0, (delta, degradation, budget)
        # measured: 1 deg -> -40.6 vs -40.9 budget; 2 deg -> -35.1 vs -34.9

    evms = [evm_db(d) for d in (0.5, 1.0, 2.0, 4.0)]
    assert all(b > a for a, b in zip(evms, evms[1:])), evms   # monotonic


def test_budget_grows_with_crest_factor(wf):
    """The budget's own physics: the error is delta * A_max / 2 per sample,
    so a higher-PAPR composite (no CFR) is more mismatch-sensitive than the
    CFR'd one -- the sensitivity outphasing trades for its linearity."""
    x_raw = wf.x
    x_cfr = cfr_clip_filter(wf.x, CFR_DB, wf.fs, wf.bw)
    b_raw = phase_mismatch_evm_budget_db(
        x_raw, outphasing_decompose(x_raw, np.abs(x_raw).max())[1], 1.0, wf.fs, wf.bw)
    b_cfr = phase_mismatch_evm_budget_db(
        x_cfr, outphasing_decompose(x_cfr, np.abs(x_cfr).max())[1], 1.0, wf.fs, wf.bw)
    assert b_raw > b_cfr + 0.5


# ---------------------------------------------- 4. efficiency laws
def test_isolated_efficiency_is_eta_pa_times_mean_cos2_and_chireix_beats_it(ideal, wf):
    """Acceptance 3.  Wilkinson: PAs never see the load change, so the DC
    draw is constant and eta_avg = eta_pa * E[cos^2 theta] exactly.  Chireix
    load-modulates the same PAs and must do better on the same waveform."""
    polar, dpa = ideal
    r_iso = OutphasingTX(polar, _lossless("isolated")).run(wf, noise=False, seed=1)
    r_chi = OutphasingTX(polar, _lossless("chireix")).run(wf, noise=False, seed=1)
    eta_pa = float(np.mean(dpa.efficiency(r_iso.taps[0].env_code)))
    expected = eta_pa * np.mean(np.cos(r_iso.theta) ** 2)
    assert r_iso.avg_efficiency(dpa)["eta_avg"] == pytest.approx(expected, rel=1e-9)
    assert r_chi.avg_efficiency(dpa)["eta_avg"] > r_iso.avg_efficiency(dpa)["eta_avg"] * 2
    # measured on this burst: isolated 10.9 %, chireix 43.9 %, eta_pa 85 %

    # insertion loss scales the delivered power, hence the average, linearly
    lossy = OutphasingCombiner(mode="isolated", combiner_loss_db=1.0)
    r_loss = OutphasingTX(polar, lossy).run(wf, noise=False, seed=1)
    assert r_loss.avg_efficiency(dpa)["eta_avg"] == pytest.approx(
        expected * 10 ** (-0.1), rel=1e-9)


def test_chireix_power_factor_peaks_at_the_compensation_angles():
    c = OutphasingCombiner(mode="chireix", chireix_theta_c_deg=60.0)
    th = np.deg2rad([0.0, 30.0, 60.0, 89.0])
    pf = c.power_factor(th)
    assert pf[2] == pytest.approx(1.0)               # theta_c
    assert pf[1] == pytest.approx(1.0)               # 90 - theta_c: the twin peak
    assert 0.9 < pf[0] < 1.0                         # in-phase: slightly reactive
    assert pf[3] < 0.1                               # near cancellation: collapses
    iso = OutphasingCombiner(mode="isolated")
    assert np.allclose(iso.power_factor(th), np.cos(th) ** 2)


# ------------------------------------------- 5. the theta distribution
def test_theta_distribution_is_reported_and_tracks_papr(ideal, wf):
    """Acceptance 4.  frac_gt80 is the single number that says how much of
    the burst sits within 10 deg of cancellation; it must be reported, and
    it must rise when the CFR is removed (higher PAPR -> more time near
    the null)."""
    polar, _ = ideal
    r = OutphasingTX(polar, _lossless()).run(wf, noise=False, seed=1)
    st = r.theta_stats()
    assert set(st) == {"p50_deg", "p95_deg", "max_deg", "frac_gt80"}
    assert 0.05 < st["frac_gt80"] < 0.5              # measured 0.21 at 8.5 dB CFR
    assert st["p50_deg"] < st["p95_deg"] <= st["max_deg"] <= 90.0
    assert r.info["theta"] == st                     # also on the info dict

    raw = theta_stats(outphasing_decompose(wf.x, np.abs(wf.x).max())[2])
    assert raw["frac_gt80"] > st["frac_gt80"]        # no CFR -> worse


# ------------------------------------------ 6. no envelope path at all
def test_env_skew_has_no_effect_on_outphasing_but_wrecks_polar(ideal, wf):
    """Acceptance 5.  The branches are constant-envelope, so the AM-path
    delay the polar chain is most sensitive to has NOTHING to act on: the
    output must be bit-identical.  The polar contrast is asserted too, so
    the invariance is not the vacuous kind (a knob nobody wired)."""
    polar, dpa = ideal
    skewed = PolarTX(ChainConfig(cfr_papr_db=CFR_DB, env_floor=0.0,
                                 env_skew_s=0.5e-9), IdealPhaseModulator(), dpa)
    y0 = OutphasingTX(polar, _lossless()).run(wf, noise=False, seed=1).y
    y1 = OutphasingTX(skewed, _lossless()).run(wf, noise=False, seed=1).y
    assert np.array_equal(y0, y1)
    assert OutphasingTX(skewed).branch_tx.cfg.env_skew_s == 0.0
    e_polar_skewed = skewed.run(wf, noise=False, seed=1).evm().db
    e_polar = polar.run(wf, noise=False, seed=1).evm().db
    assert e_polar_skewed > e_polar + 10.0           # measured -21.9 vs -41.6


# ------------------------------------------------- combiner contract
def test_combiner_validates_like_doherty():
    with pytest.raises(ValueError):
        OutphasingCombiner(mode="wilkinson")
    with pytest.raises(ValueError):
        OutphasingCombiner(gain_imbalance=(0.1,))
    c = OutphasingCombiner(gain_imbalance=(0.05, -0.05), combiner_loss_db=0.4)
    assert c.combining_loss_db() == pytest.approx(-0.4)    # gain alone costs nothing
    c2 = OutphasingCombiner(phase_imbalance_deg=(0.0, 10.0), combiner_loss_db=0.0)
    assert c2.combining_loss_db() == pytest.approx(20 * np.log10(np.cos(np.deg2rad(5.0))))
    y1, y2 = np.ones(4, complex), np.ones(4, complex)
    assert np.allclose(OutphasingCombiner(combiner_loss_db=0.0).combine(y1, y2), 2.0)


# ---------------------------------------------- preset + registry
def test_preset_is_registered_as_a_standard_chain_and_reports():
    from polartx.guiutil import PRESETS, build_preset, run_chain_report
    name = "WiFi 160 MHz (outphasing)"
    assert name in PRESETS and not name.startswith("Bench:")
    p = build_preset(name, n_bits=9)                 # GUI-style override
    assert isinstance(p.tx, OutphasingTX)
    assert p.polar_tx.phasemod.cfg.n_bits == 9       # forwarded to the plan
    assert p.tx.tx is p.polar_tx                     # same chain, decomposed twice
    rep = run_chain_report(name, seed=1, n_units=3)
    assert rep["metrics"].get("mask") in ("PASS", "FAIL")
    assert "EVM [dB]" in rep["metrics"]
    import matplotlib.pyplot as plt
    plt.close(rep["fig"])


def test_preset_shares_the_frequency_plan_with_wifi_dtc():
    o = wifi_outphasing(bw=160e6, qam=1024)
    d = wifi_dtc(bw=160e6, qam=1024)
    assert o.fs_bb == d.fs_bb
    assert o.polar_tx.phasemod.cfg == d.tx.phasemod.cfg
    assert o.tx.combiner.mode == "chireix"
    r = o.tx.run(o.make_waveform(n_symbols=3, seed=0), noise=True, seed=1)
    assert r.evm_equalize_default == "scalar"        # same convention as polar
    assert r.evm().db < -30.0                         # measured -36.3 dB


# --------------------------------------- one LO for both branches (B7)
def test_branches_share_one_lo_and_the_penalty_matches_the_shared_lo_budget():
    """The two DTC branches hang off one PLL, so their LO phase noise is
    common: the outphasing floor over the polar floor must then be the
    selector's shared-LO budget (independent floors +10log10(PAPR/2) only,
    the synth term unchanged) rather than the independent-LO one, which
    the stage-1 chain simulated and which was ~5 dB too pessimistic.  The
    CFR floor is subtracted in power so only the technology floors are
    compared, on the same 4-symbol burst."""
    from polartx import CartesianTX, RFDAC, RFDACConfig
    from polartx.selector import Requirement, select
    p = wifi_dtc(bw=160e6, qam=1024)
    wf4 = p.make_waveform(n_symbols=4, seed=0)
    x = cfr_clip_filter(wf4.x, CFR_DB, wf4.fs, wf4.bw)
    papr = float(10 * np.log10(np.abs(x).max() ** 2 / np.mean(np.abs(x) ** 2)))
    floor = CartesianTX(ChainConfig(cfr_papr_db=CFR_DB),
                        RFDAC(RFDACConfig(n_bits=14))).run(wf4, noise=False).evm().db

    def floors_only(evm_db):
        return 10 * np.log10(10 ** (evm_db / 10) - 10 ** (floor / 10))

    o = OutphasingTX(p.tx, OutphasingCombiner(mode="chireix"))
    assert o.shared_lo
    r = o.run(wf4, noise=True, seed=1)
    assert r.info["shared_lo"] is True
    gap = floors_only(r.evm().db) - floors_only(p.tx.run(wf4, noise=True, seed=1).evm().db)
    kw = dict(bw_hz=160e6, modulation="ofdm", fout=5.9e9, dtc_bits=11, papr_db=papr,
              dtc_inl_floor_db=float("-inf"), branch_phase_mismatch_deg=0.0,
              synth_loop_bw=400e3)
    def budget(shared):
        d = {c.arch: c for c in select(Requirement("w", outphasing_shared_lo=shared,
                                                    **kw)).candidates}
        return d["outphasing"].evm_db - d["dtc_open_loop"].evm_db
    assert abs(gap - budget(True)) < 1.5, (gap, budget(True))    # -0.3 vs 0.44 (8 sym: 0.4)
    assert gap < budget(False) - 1.5                              # not the 5.9 dB case
    # the two branch modulators are the base one with the LO pinned; the
    # base chain and its calibration state are untouched
    assert o.branch_tx.phasemod is p.tx.phasemod
    assert o.branch_tx.phasemod.cfg.lo_pn_seed is None
    assert o._branch_chain(5).phasemod.cfg.lo_pn_seed == 5


def test_no_lo_model_means_nothing_to_share(ideal):
    polar, _ = ideal
    o = OutphasingTX(polar, _lossless())
    assert not o.shared_lo
    assert o._branch_chain(3) is o.branch_tx
