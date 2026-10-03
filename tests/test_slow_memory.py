"""Slow-state residual memory from a complete measured source (stage 2).

Everything here runs on the VIRTUAL self-heating DUT
(``measured.synthetic_thermal_source``, vendored ThermalReferencePA with
taus 5 / 30 us), so none of it needs data and none of it skips: this is
the always-on coverage of the slow-state path.  The four acceptance items
of the design note are the first four tests; the rest pin the chain
contract and the conventions the numbers depend on.
"""
import numpy as np
import pytest

from polartx.chain import ChainConfig, PolarTX
from polartx.measured import (SYNTHETIC_THERMAL_TAUS_S, _fit_on_pair,
                              dpa_from_complete_source, extract_polar_characteristics,
                              fast_spline_residual, fit_residual_memory,
                              identify_slow_states, residual_training_pair,
                              state_spline_residual, static_prediction,
                              synthetic_thermal_source)
from polartx.phasemod import IdealPhaseModulator
from polartx.vendor.padpd.pa import load_model, nmse_db
from polartx.waveforms.base import Waveform


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """Written to disk and read back: the path a recorded source takes."""
    path = tmp_path_factory.mktemp("src") / "thermal_source.npz"
    return synthetic_thermal_source(str(path))


@pytest.fixture(scope="module")
def fitted(source):
    return dpa_from_complete_source(source)


# ------------------------------------------- 1. tau identification
def test_offline_tau_identification_recovers_the_true_constants(source):
    """Acceptance: the step capture alone, offline, gives the DUT's two
    thermal constants within 10 % (PA_DPD's own demo: 5.1 / 29.6 us)."""
    gm = identify_slow_states(source)
    taus = gm.taus_heat_s
    assert len(taus) == 2
    for got, true in zip(taus, SYNTHETIC_THERMAL_TAUS_S):
        assert abs(got - true) / true < 0.10, (got, true)   # 5.14 / 29.64 us
    alphas = gm.state_alphas(source["fs"])
    assert np.allclose(alphas, np.exp(-1.0 / (np.asarray(taus) * source["fs"])))


# --------------------------------- 2. slow state on the burst capture
def test_state_residual_beats_the_fast_only_residual_on_the_burst(fitted):
    """Acceptance: on the burst validation tail (heating AND cooling
    edges), the state-conditioned residual is >= 6 dB better than the
    SplineGMP residual fitted to the same pair without states."""
    _, ch = fitted
    rep = ch["memory_report"]
    assert rep["model_class"] == "StateConditionedSpline"
    assert rep["fast_only_model_class"] == "SplineGMP"
    assert rep["state_gain_db"] >= 6.0, rep                    # measured 13.3
    assert rep["residual_nmse_db"] < -28.0                      # measured -37.0
    assert rep["fast_only_nmse_db"] < rep["memoryless_nmse_db"] < rep["static_nmse_db"]
    # every stage of the ladder buys something on a slow-state device
    assert abs(rep["residual_train_nmse_db"] - rep["residual_nmse_db"]) < 3.0


# --------------------------------- 3. stationary control
def test_states_buy_nothing_on_a_stationary_capture(fitted):
    """Acceptance: on the warm, long main capture the slow state is
    constant -- unobservable -- so the state model must not look better
    than the stateless one (<= 1 dB either way).  Same alphas, same
    split, same fitter as the burst fit."""
    _, ch = fitted
    alphas = ch["memory_report"]["state_alphas"]
    u, v = residual_training_pair(ch)
    _, rs = _fit_on_pair(u, v, lambda ut: state_spline_residual(ut, alphas),
                         0.6, 1e-9, ch)
    _, rf = _fit_on_pair(u, v, fast_spline_residual, 0.6, 1e-9, ch)
    assert abs(rs["residual_nmse_db"] - rf["residual_nmse_db"]) <= 1.0, (
        rs["residual_nmse_db"], rf["residual_nmse_db"])         # -41.46 vs -41.90


def test_the_virtual_dut_is_one_continuous_device_when_the_heat_is_frozen(fitted):
    """With heat_gain=0 the thermal DUT must BE its ReferencePA: block-wise
    processing may change only the Saleh stage's parameters, never the
    signal path.  Until PA_DPD 08b9725 every 128-sample block went through
    a fresh ReferencePA whose FIRs restarted from rest, so in-block samples
    0/1/2 were wrong and every number measured on this DUT sat on a
    -34.9 dB floor (the stationary fast-only residual landed at -34.0).
    This pins the floor's absence, not a looser threshold on it: the two
    calls agree sample by sample to rounding, block edges included, and
    the residual models now go well past where the floor was."""
    from polartx.vendor.padpd.pa.thermal import ThermalReferencePA
    from polartx.vendor.padpd.waveform.ofdm import OFDMConfig, generate_ofdm
    fs = 80e6
    x = generate_ofdm(OFDMConfig(bandwidth_hz=fs / 4, qam_order=1024,
                                 n_symbols=16, seed=0)).x
    pa = ThermalReferencePA(drive0=0.13, fs=fs, heat_gain=0.0)
    y_blocks = pa(x)
    y_cont = pa._drift.pa()(x)
    # not bit-exact: the gain normalisation now sits before FIR_out
    # (measured max error 8.9e-16, 3e-16 of full scale)
    tol = 1e-12 * np.max(np.abs(y_cont))
    err = np.abs(y_blocks - y_cont)
    assert np.all(err <= tol), err.max()
    offset = np.arange(x.size) % pa.block
    assert np.all(err[offset < 3] <= tol)                      # was -14/-25/-50 dB
    # and the stationary fast-only residual is no longer held at ~-35 dB
    _, ch = fitted
    u, v = residual_training_pair(ch)
    _, rf = _fit_on_pair(u, v, fast_spline_residual, 0.6, 1e-9, ch)
    assert rf["residual_nmse_db"] < -38.0                       # measured -41.9 (was -34.0)


def test_a_cold_short_main_capture_is_not_stationary():
    """Why the control needs a WARM capture: recorded cold, the 'main'
    capture is itself a heating transient and the state model looks
    several dB better on it -- the trap the synthetic source avoids.
    (Measured: cold 54 us capture, state 7.6 dB better; 5.4 dB
    before the PA_DPD 08b9725 DUT fix.)"""
    from polartx.vendor.padpd.pa.thermal import ThermalReferencePA
    from polartx.vendor.padpd.waveform.ofdm import OFDMConfig, generate_ofdm
    fs = 80e6
    x = generate_ofdm(OFDMConfig(bandwidth_hz=fs / 4, qam_order=1024,
                                 n_symbols=4, seed=0)).x
    pa = ThermalReferencePA(drive0=0.13, taus_s=SYNTHETIC_THERMAL_TAUS_S, fs=fs)
    assert pa.state == 0.0
    y = pa(x)
    assert pa.state > 0.3                                       # it heated: 0 -> 0.61
    ch = extract_polar_characteristics(x, y)
    u, v = residual_training_pair(ch)
    gm_alphas = np.exp(-1.0 / (np.asarray(SYNTHETIC_THERMAL_TAUS_S) * fs))
    _, rs = _fit_on_pair(u, v, lambda ut: state_spline_residual(ut, gm_alphas),
                         0.6, 1e-9, ch)
    _, rf = _fit_on_pair(u, v, fast_spline_residual, 0.6, 1e-9, ch)
    assert rf["residual_nmse_db"] - rs["residual_nmse_db"] > 3.0


# --------------------------------- 4. missing groups are named
@pytest.mark.parametrize("drop", [("burst",), ("step",), ("burst", "step")])
def test_missing_capture_groups_raise_and_say_which(fitted, source, drop):
    _, ch = fitted
    stripped = dict(source, extras={k: v for k, v in source["extras"].items()
                                    if k not in drop})
    with pytest.raises(ValueError) as e:
        fit_residual_memory(ch, source=stripped)
    msg = str(e.value)
    for g in drop:
        assert repr(g) in msg
    for g in {"burst", "step"} - set(drop):
        assert f"{g!r} =" not in msg                            # only the missing ones


def test_a_step_capture_without_modulation_is_refused(source):
    """No significant gain modulation -> no slow state to model; the
    path refuses rather than fitting states to noise."""
    xs = source["extras"]["step"]["x"]
    flat = dict(source, extras=dict(source["extras"],
                                    step={"x": xs, "y": 0.8 * xs}))
    with pytest.raises(ValueError, match="no significant gain modulation"):
        identify_slow_states(flat)


# --------------------------------- the chain contract
def test_chain_with_state_memory_reproduces_the_burst_and_round_trips(fitted, source, tmp_path):
    """Built as documented (fs_scale_fixed = capture full scale, capture
    sample rate, one run = cold start), the chain driven with the burst
    input reproduces the measured burst output to the model's own
    validation accuracy; and the model survives save -> load_model
    bit for bit in the chain (same process, same BLAS)."""
    dpa, ch = fitted
    rep = ch["memory_report"]
    b = source["extras"]["burst"]
    wf = Waveform(x=b["x"], fs=source["fs"], bw=source["fs"] / 4, kind="ofdm")
    cfg = ChainConfig(fs_scale_fixed=ch["fs_scale"])
    val = slice(rep["n_train"], None)
    g = ch["chain_gain"]
    y0 = PolarTX(cfg, IdealPhaseModulator(), dpa).run(wf, noise=False).y
    assert nmse_db(static_prediction(ch, b["x"])[val] / g, y0[val]) < -35.0   # -39.9
    tx = PolarTX(cfg, IdealPhaseModulator(), dpa, memory=ch["memory"])
    y1 = tx.run(wf, noise=False).y
    chain_nmse = nmse_db(b["y"][val] / g, y1[val])
    assert abs(chain_nmse - rep["residual_nmse_db"]) < 1.0      # -37.2 vs -37.0
    path = tmp_path / "state.npz"
    ch["memory"].save(str(path))
    back = load_model(str(path))
    y2 = PolarTX(cfg, IdealPhaseModulator(), dpa, memory=back).run(wf, noise=False).y
    assert np.array_equal(y1, y2)


def test_report_states_the_conditioning_honestly(fitted):
    """The state model is rank deficient by construction (partition of
    unity in its additive and interaction blocks) and has dead columns
    (one q_scale for all states); the report says so instead of printing
    a condition number of 1e18."""
    _, ch = fitted
    rep, m = ch["memory_report"], ch["memory"]
    structural = m.n_states * m.memory_depth + (m.n_states * m.n_basis if m.interaction else 0)
    assert rep["rank_deficiency"] >= structural                 # 49 >= 28
    assert rep["dead_columns"] > 0                              # 14
    assert rep["condition_number"] < 1e8                        # 1.6e4 on the spanned space
    assert rep["n_coeffs"] == m.n_coeffs == 180


def test_source_meta_and_constants_travel(fitted, source):
    _, ch = fitted
    assert ch["fs"] == source["fs"] == 80e6
    assert ch["source_meta"]["true_taus_s"] == list(SYNTHETIC_THERMAL_TAUS_S)
    rep = ch["memory_report"]
    assert rep["source"] == "complete:burst" and rep["split"] == 0.6
    assert len(rep["state_alphas"]) == len(rep["taus_heat_s"]) == 2
