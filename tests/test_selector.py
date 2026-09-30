"""Architecture selector: the narrowband/wideband split, the two new
topologies' floors, and the analytic laws against the chains they screen.

The selector is an analytic screen; what makes it trustworthy is that its
closed forms reproduce what the simulated chains measure on the same burst.
The second half of this file does exactly that comparison, on the 4-symbol
WiFi 160 MHz 1024-QAM burst the topology tests use, with the chain's own CFR
floor subtracted in power so only the technology floors are compared.
"""
import warnings

import numpy as np
import pytest

from polartx import (CartesianTX, ChainConfig, OutphasingCombiner, OutphasingTX,
                     RFDAC, RFDACConfig, edr_dpsk, wifi_dtc, wifi_outphasing,
                     wifi_rfdac)
from polartx.outphasing import outphasing_decompose, phase_mismatch_evm_budget_db
from polartx.selector import (_DPSK_PAPR_DB, ARCH_LABELS, Requirement,
                              amplitude_distribution, eta_outphasing_chireix,
                              eta_polar_scpa, eta_rfdac_iq_cells,
                              outphasing_branch_penalty_db,
                              outphasing_mismatch_evm_db, rfdac_quant_evm_db,
                              select)
from polartx.vendor.padpd.cfr import cfr_clip_filter

ARCHS = {"adpll_two_point", "dtc_open_loop", "outphasing", "rfdac_cartesian"}


def _by_arch(rep):
    return {c.arch: c for c in rep.candidates}


# ------------------------------------------------- the original split
def test_wideband_excludes_adpll():
    """Above the two-point coverage ceiling the ADPLL is infeasible and the
    open-loop DTC is recommended (it meets the target with the best
    efficiency of the bandwidth-agnostic three)."""
    for bw in (80e6, 200e6, 320e6):
        rep = select(Requirement("wb", bw, "ofdm", fout=6e9))
        adpll = next(c for c in rep.candidates if c.arch == "adpll_two_point")
        assert not adpll.feasible
        assert rep.best.arch == "dtc_open_loop"


def test_narrowband_prefers_adpll_when_calibrated():
    """With a calibrated two-point (0.2%), the in-loop-FM ADPLL beats the
    open-loop DTC across the narrowband range — no DTC quant/jitter/INL floor.
    Same efficiency law as the DTC (same DPA), so the EVM tie-break decides."""
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
    tbl = rep.table()
    for arch in ARCHS:
        assert arch in tbl


def test_infeasible_everything_reports_no_recommendation():
    """A ceiling below the bandwidth knocks the ADPLL out; the rest stay."""
    rep = select(Requirement("x", 100e6, "ofdm", adpll_bw_ceiling=10e6))
    assert rep.best.arch == "dtc_open_loop"
    assert rep.recommendation


# ---------------------------------------------- stage 3: four candidates
def test_four_candidates_are_scored_with_terms_and_efficiency():
    rep = select(Requirement("w", 160e6, "ofdm", fout=5.9e9))
    d = _by_arch(rep)
    assert set(d) == ARCHS == set(ARCH_LABELS)
    for c in d.values():
        assert 0.0 < c.eta_avg <= 1.0 and c.terms and c.notes
        assert "synth_pn" in c.terms or "synth_pn+branches" in c.terms
    assert {"dtc_quant+branches", "branch_mismatch"} <= set(d["outphasing"].terms)
    assert {"dac_quant", "dac_jitter", "iq_image"} <= set(d["rfdac_cartesian"].terms)


def test_ble_still_picks_the_adpll():
    """Acceptance: a BLE requirement keeps its verdict with four candidates.
    Constant envelope: outphasing gains nothing (θ ≡ 0, mismatch term
    vanishes) and the RF-DAC pays its |I|+|Q| law for no linearity need."""
    rep = select(Requirement("BLE-1M", 1e6, "gfsk", evm_db_max=-20,
                             constant_envelope=True, fout=2.44e9))
    assert rep.best.arch == "adpll_two_point"
    assert rep.suggest_preset().startswith("ble_adpll")
    d = _by_arch(rep)
    assert d["outphasing"].terms["branch_mismatch"] == float("-inf")
    assert d["outphasing"].eta_avg < d["adpll_two_point"].eta_avg   # 71 vs 85 %
    assert d["rfdac_cartesian"].eta_avg == pytest.approx(0.85 * np.pi / 4, rel=1e-6)


def test_wifi7_320_4096qam_does_not_pick_outphasing():
    """Acceptance: outphasing is never the answer for 320 MHz 4096-QAM.  Its
    floors are the DTC's plus the branch penalty plus the mismatch term, so
    it ranks below the DTC on EVM and on efficiency alike."""
    rep = select(Requirement("WiFi7-320", 320e6, "ofdm", evm_db_max=-38,
                             fout=6e9, dtc_bits=12))
    assert rep.best.arch != "outphasing"
    d = _by_arch(rep)
    assert d["outphasing"].evm_db > d["dtc_open_loop"].evm_db + 3.0   # measured +7.3
    assert d["outphasing"].eta_avg < d["dtc_open_loop"].eta_avg
    order = [c.arch for c in rep.ranked()]
    assert order.index("outphasing") > order.index("dtc_open_loop")


def test_ranking_rule_efficiency_decides_once_the_target_is_met():
    """The stage-3 rule: among candidates that meet the target the most
    efficient wins; a candidate that misses it never outranks one that
    meets it, however good its efficiency."""
    easy = select(Requirement("w", 160e6, "ofdm", evm_db_max=-35, fout=5.9e9))
    assert easy.best.arch == "dtc_open_loop"            # all meet; 44 % > 28 %
    d = _by_arch(easy)
    assert d["rfdac_cartesian"].meets(-35) and d["rfdac_cartesian"].evm_db < 0
    # cripple the DTC (6 bits): it misses -40, the RF-DAC still meets it
    hard = select(Requirement("w", 160e6, "ofdm", evm_db_max=-40, fout=5.9e9,
                              dtc_bits=6))
    h = _by_arch(hard)
    assert not h["dtc_open_loop"].meets(-40) and h["rfdac_cartesian"].meets(-40)
    assert hard.best.arch == "rfdac_cartesian"
    assert "wifi_rfdac" in hard.suggest_preset()
    assert "Cartesian RF-DAC" in hard.recommendation


def test_efficiency_gate_excludes_with_a_reason():
    rep = select(Requirement("w", 160e6, "ofdm", fout=5.9e9, eta_avg_min=0.35))
    d = _by_arch(rep)
    assert not d["rfdac_cartesian"].feasible
    assert "efficiency" in d["rfdac_cartesian"].notes[-1]
    assert d["dtc_open_loop"].feasible and rep.best.arch == "dtc_open_loop"
    assert "rfdac_cartesian excluded" in rep.recommendation


def test_shared_lo_is_worth_the_branch_penalty_on_the_synth_term():
    kw = dict(bw_hz=160e6, modulation="ofdm", fout=5.9e9)
    shared = _by_arch(select(Requirement("s", outphasing_shared_lo=True, **kw)))
    indep = _by_arch(select(Requirement("s", outphasing_shared_lo=False, **kw)))
    pen = outphasing_branch_penalty_db(8.5)
    assert pen == pytest.approx(10 * np.log10(10 ** 0.85 / 2))       # +5.5 dB
    assert indep["outphasing"].terms["synth_pn+branches"] == pytest.approx(
        shared["outphasing"].terms["synth_pn"] + pen)
    assert indep["outphasing"].evm_db > shared["outphasing"].evm_db
    assert outphasing_branch_penalty_db(0.0) == pytest.approx(-10 * np.log10(2))


def test_gui_report_carries_four_rows_and_efficiency():
    from polartx.guiutil import run_selector_report
    r = run_selector_report(bw_hz=160e6, evm_db_max=-35.0)
    for arch in ARCHS:
        assert arch in r["metrics"]
    assert "eta" in r["metrics"]["dtc_open_loop"]
    assert len(r["fig"].axes[0].lines) >= 4
    import matplotlib.pyplot as plt
    plt.close(r["fig"])


# ------------------------------------- the laws against the chains
@pytest.fixture(scope="module")
def burst():
    p = wifi_dtc(bw=160e6, qam=1024)
    wf = p.make_waveform(n_symbols=4, seed=0)
    x = cfr_clip_filter(wf.x, 8.5, wf.fs, wf.bw)
    papr = float(10 * np.log10(np.abs(x).max() ** 2 / np.mean(np.abs(x) ** 2)))
    floor = CartesianTX(ChainConfig(cfr_papr_db=8.5),
                        RFDAC(RFDACConfig(n_bits=14))).run(wf, noise=False).evm().db
    return p, wf, x, papr, floor


def _floor_only(evm_db, floor_db):
    return float(10 * np.log10(10 ** (evm_db / 10) - 10 ** (floor_db / 10)))


def test_amplitude_distribution_is_a_clipped_rayleigh_at_the_stated_papr():
    for papr in (8.5, 10.0):
        a, w = amplitude_distribution(papr)
        assert w.sum() == pytest.approx(1.0)
        assert np.sum(w * a * a) * 10 ** (papr / 10) == pytest.approx(1.0, rel=1e-2)
    # at a low PAPR the clipped tail is a sizeable mass at the peak and the
    # distribution's own PAPR exceeds the parameter (3.4 -> 3.9 dB): known
    a, w = amplitude_distribution(3.4)
    assert 0.85 < np.sum(w * a * a) * 10 ** 0.34 < 0.95            # measured 0.888
    a, w = amplitude_distribution(0.0)
    assert a.tolist() == [1.0] and w.tolist() == [1.0]


def test_dpsk_papr_constant_matches_the_edr_waveform():
    x = edr_dpsk(2000, 8e6).x
    papr = 10 * np.log10(np.abs(x).max() ** 2 / np.mean(np.abs(x) ** 2))
    assert abs(papr - _DPSK_PAPR_DB) < 0.5                # measured 3.2-3.5


def test_efficiency_laws_reproduce_the_three_chains(burst):
    """Each topology's analytic average efficiency at the burst's own PAPR
    must land on what the chain measures on that burst."""
    p, wf, _, papr, _ = burst
    req = Requirement("w", 160e6, "ofdm", papr_db=papr)
    o = wifi_outphasing(bw=160e6, qam=1024)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = wifi_rfdac(bw=160e6, qam=1024)
    meas = {
        "polar": p.tx.run(wf, noise=False).avg_efficiency(p.tx.dpa)["eta_avg"],
        "outph": o.tx.run(wf, noise=False).avg_efficiency(p.tx.dpa)["eta_avg"],
        "rfdac": r.tx.run(wf, noise=False).avg_efficiency()["eta_avg"],
    }
    ana = {"polar": eta_polar_scpa(req), "outph": eta_outphasing_chireix(req),
           "rfdac": eta_rfdac_iq_cells(req)}
    for k in meas:
        assert abs(ana[k] - meas[k]) < 0.03, (k, ana[k], meas[k])
        # measured on the 8-symbol burst: .428/.400/.280 vs .428/.402/.270
    assert ana["polar"] > ana["outph"] > ana["rfdac"]


def test_outphasing_mismatch_term_matches_the_chain_budget(burst):
    """The closed form (with its measured in-band share) must agree with the
    waveform-level budget the outphasing tests are pinned to, at 1 and 2
    degrees, and slope 6 dB per octave of the error."""
    _, wf, x, papr, _ = burst
    _, s2, _ = outphasing_decompose(x, float(np.abs(x).max()))
    for d in (1.0, 2.0):
        ana = outphasing_mismatch_evm_db(d, papr, Requirement("w", 1e6).quad_inband_db)
        bud = phase_mismatch_evm_budget_db(x, s2, d, wf.fs, wf.bw)
        assert abs(ana - bud) < 1.5, (d, ana, bud)      # 1 deg: -40.8 vs -40.9
    one, two = (outphasing_mismatch_evm_db(d, papr, -7.5) for d in (1.0, 2.0))
    assert two - one == pytest.approx(20 * np.log10(2), abs=1e-9)


def test_rfdac_quantization_term_matches_the_chain(burst):
    """6-bit I/Q arrays on the chain, CFR floor subtracted in power, against
    the closed form 2Δ²/12 spread over the oversampling ratio."""
    _, wf, _, papr, floor = burst
    e6 = CartesianTX(ChainConfig(cfr_papr_db=8.5),
                     RFDAC(RFDACConfig(n_bits=6))).run(wf, noise=False).evm().db
    meas = _floor_only(e6, floor)
    ana = rfdac_quant_evm_db(6, papr, 4.0)
    assert abs(meas - ana) < 1.0, (meas, ana)             # -41.5 vs -41.1


def test_technology_floors_track_the_three_chains(burst):
    """Noise on, floors only: the analytic RF-DAC and polar floors are equal
    (same LO, both LO-limited) and the chains agree; the outphasing penalty
    over polar matches the chain, whose two branches share one LO
    (``outphasing_shared_lo=True``, the default)."""
    p, wf, _, papr, floor = burst
    o = wifi_outphasing(bw=160e6, qam=1024)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = wifi_rfdac(bw=160e6, qam=1024)
    meas = {k: _floor_only(t.run(wf, noise=True, seed=1).evm().db, floor)
            for k, t in (("polar", p.tx), ("outph", o.tx), ("rfdac", r.tx))}
    req = Requirement("w", 160e6, "ofdm", fout=5.9e9, dtc_bits=11, dac_bits=12,
                      papr_db=papr, dtc_inl_floor_db=float("-inf"),
                      iq_gain_err_db=0.0, iq_phase_err_deg=0.0,
                      branch_phase_mismatch_deg=0.0, outphasing_shared_lo=True,
                      synth_loop_bw=400e3)
    d = _by_arch(select(req))
    ana = {"polar": d["dtc_open_loop"].evm_db, "outph": d["outphasing"].evm_db,
           "rfdac": d["rfdac_cartesian"].evm_db}
    assert abs((ana["rfdac"] - ana["polar"]) - (meas["rfdac"] - meas["polar"])) < 1.0
    assert abs((ana["outph"] - ana["polar"]) - (meas["outph"] - meas["polar"])) < 2.0
    # measured (4 symbols, shared LO): polar -43.1, outphasing -43.4, RF-DAC -43.1
    for k in meas:
        assert abs(ana[k] - meas[k]) < 5.0, (k, ana[k], meas[k])


def test_outphasing_and_rfdac_chains_are_what_the_selector_suggests():
    """suggest_preset names a preset that exists and builds."""
    import polartx
    for arch, req in (("outphasing", Requirement("w", 160e6, "ofdm")),
                      ("rfdac_cartesian", Requirement("w", 160e6, "ofdm"))):
        rep = select(req)
        rep.candidates = [c for c in rep.candidates if c.arch == arch]
        name = rep.suggest_preset().split("(")[0]
        assert callable(getattr(polartx, name))
    tx = OutphasingTX(wifi_dtc().tx, OutphasingCombiner(mode="chireix"))
    assert tx.combiner.chireix_theta_c_deg == Requirement("w", 1e6).chireix_theta_c_deg
