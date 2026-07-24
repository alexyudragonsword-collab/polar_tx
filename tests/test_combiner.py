"""Multi-core / Doherty power combiner: derived signal + efficiency."""
import numpy as np
import pytest

from polartx.dpa import DPA, DPAConfig, DohertyCombiner, imbalance_montecarlo


def test_balanced_reproduces_target_amplitude():
    """An ideal, balanced, lossless combiner outputs exactly x."""
    c = DohertyCombiner(n_way=2, backoff_db=6.0, combiner_loss_db=0.0)
    x = np.linspace(0, 1, 257)
    assert np.max(np.abs(np.abs(c.combine(x)) - x)) < 1e-12


def test_ideal_classB_efficiency_peaks_at_backoff_and_full():
    """Ideal class-B: eta rises to pi/4 at the backoff point, flat above."""
    c = DohertyCombiner(n_way=2, backoff_db=6.0, eta_peak=np.pi / 4,
                        combiner_loss_db=0.0)
    v_t = 10 ** (-6.0 / 20.0)
    assert c.efficiency(np.array([v_t]))[0] == pytest.approx(np.pi / 4, rel=1e-3)
    assert c.efficiency(np.array([1.0]))[0] == pytest.approx(np.pi / 4, rel=1e-3)
    # halfway to backoff -> about half the peak (linear rise)
    assert c.efficiency(np.array([v_t / 2]))[0] == pytest.approx(np.pi / 8, rel=1e-2)


def test_classC_peaking_creates_the_double_hump_dip():
    """Class-C peaking dips between the two efficiency peaks."""
    c = DohertyCombiner(n_way=2, backoff_db=6.0, peaking="C", dip=0.12,
                        combiner_loss_db=0.0)
    v_t = 10 ** (-6.0 / 20.0)
    mid = (v_t + 1.0) / 2.0
    eta_mid = c.efficiency(np.array([mid]))[0]
    eta_pk = c.efficiency(np.array([1.0]))[0]
    assert eta_mid < eta_pk                 # there is a dip
    assert eta_mid > 0.6 * eta_pk           # but a shallow one


def test_gain_imbalance_is_not_a_combining_loss():
    """Pure gain imbalance is an absorbable gain error, not lost power;
    only phase misalignment (and insertion loss) reduce combined power."""
    gain_only = DohertyCombiner(n_way=2, gain_imbalance=(0.0, 0.15),
                                combiner_loss_db=0.0)
    phase_only = DohertyCombiner(n_way=2, phase_imbalance_deg=(0.0, 20.0),
                                 combiner_loss_db=0.0)
    assert abs(gain_only.combining_loss_db()) < 1e-9
    assert phase_only.combining_loss_db() < -0.05


def test_imbalance_creates_amam_ampm_distortion():
    """Core imbalance shows up as AM-AM ripple and AM-PM."""
    clean = DohertyCombiner(n_way=2).am_curves()
    dirty = DohertyCombiner(n_way=2, gain_imbalance=(0.0, 0.1),
                            phase_imbalance_deg=(0.0, 8.0)).am_curves()
    assert clean["amam_ripple_db"] < 1e-6 and clean["ampm_pp_deg"] < 1e-6
    assert dirty["amam_ripple_db"] > 0.1 and dirty["ampm_pp_deg"] > 1.0


def test_insertion_loss_lowers_efficiency():
    lossless = DohertyCombiner(combiner_loss_db=0.0).efficiency(np.array([1.0]))[0]
    lossy = DohertyCombiner(combiner_loss_db=1.0).efficiency(np.array([1.0]))[0]
    assert lossy < lossless
    assert lossy / lossless == pytest.approx(10 ** (-1.0 / 10.0), rel=1e-6)


def test_three_way_has_two_turn_on_points():
    c = DohertyCombiner(n_way=3, backoff_db=9.0)
    assert c._vt.size == 2
    assert c._vt[0] == pytest.approx(10 ** (-9.0 / 20.0), rel=1e-6)
    # still balanced-exact at the ideal
    x = np.linspace(0, 1, 129)
    assert np.max(np.abs(np.abs(
        DohertyCombiner(n_way=3, combiner_loss_db=0.0).combine(x)) - x)) < 1e-12


def test_plugs_into_dpa_characteristics():
    """The combiner supplies AM-AM / AM-PM / efficiency to a DPA."""
    specs = DohertyCombiner(n_way=2, backoff_db=6.0, peaking="C").to_dpa_specs()
    dpa = DPA(DPAConfig(n_bits=10, amam=specs["amam"],
                        ampm_lut=specs["ampm_lut"], eff=specs["eff"]))
    code = dpa.encode(np.linspace(0, 1, 500))
    ae = dpa.average_efficiency(code)
    assert 0.0 < ae["eta_avg"] < 1.0
    # full-scale amplitude normalizes to 1
    assert abs(dpa.amp_table[-1]) == pytest.approx(1.0, rel=1e-6)


def test_montecarlo_yield_monotonic_in_sigma():
    """Tighter matching -> lower AM-AM ripple / AM-PM spread (p95)."""
    tight = imbalance_montecarlo(DohertyCombiner(n_way=2), sigma_gain=0.01,
                                 sigma_phase_deg=1.0, n_trials=200, seed=1)
    loose = imbalance_montecarlo(DohertyCombiner(n_way=2), sigma_gain=0.05,
                                 sigma_phase_deg=6.0, n_trials=200, seed=1)
    assert tight["ampm_pp_deg"]["p95"] < loose["ampm_pp_deg"]["p95"]
    assert tight["amam_ripple_db"]["p95"] < loose["amam_ripple_db"]["p95"]
