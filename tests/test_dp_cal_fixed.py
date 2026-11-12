"""Fixed-point sign-sign LMS for the two-point gain (cal/dp_cal_fixed.py).

Same scenario as test_dp_cal_online.py (BLE, event engine, 5 % direct-path
gain error).  The float calibrator is the reference the fixed one answers
to; the integer streams are the golden vectors an RTL is checked against.
"""
import numpy as np
import pytest

from polartx.cal.dp_cal_fixed import (DpCalFixedConfig, FixedSignSignLMS,
                                      golden_trace, step_codes)
from polartx.export.dp_cal_vectors import (DP_GAIN, N_BITS, NOISE_SEED, WAVEFORM_SEED,
                                          run_scenario, write_vectors)
from polartx.presets import ble_1m_adpll
from polartx.vendor.pllsim.calibration.lms import SignSignLMS


def _tail(trace):
    trace = np.asarray(trace, dtype=float)
    return float(trace[-trace.size // 10:].mean())


@pytest.fixture(scope="module")
def fixed_run():
    return run_scenario()


@pytest.fixture(scope="module")
def float_run():
    p = ble_1m_adpll(mode="event", dp_gain=DP_GAIN)
    p.tx.phasemod.dp_cal = SignSignLMS(init=DP_GAIN, mu=2e-5)
    wf = p.make_waveform(n_bits=N_BITS, seed=WAVEFORM_SEED)
    res = p.tx.run(wf, noise=True, seed=NOISE_SEED)
    return res.info["phasemod"]["sim"].cal_traces["dp_gain"], res


def test_fixed_converges_like_float(fixed_run, float_run):
    cal, res = fixed_run
    tr = res.info["phasemod"]["sim"].cal_traces["dp_gain"]
    assert np.array_equal(tr, np.asarray(cal.v_codes) / 2 ** cal.cfg.v_frac)
    tail, evm = _tail(tr), res.evm()["evm_pct"]
    assert abs(tail - 1.0) < 0.01                 # the float test's bound
    assert 2.5 < evm < 3.8                         # matched floor, not the 5 % ~6.5 %
    ftr, fres = float_run
    # fixed answers to float: same convergence point, same EVM class
    assert abs(tail - _tail(ftr)) < 0.005
    assert abs(evm - fres.evm()["evm_pct"]) < 0.5


def test_engine_streams_reproduce_the_golden_trace(fixed_run):
    cal, _ = fixed_run
    g = golden_trace(cal.e_codes, cal.r_codes, cal.v_start, cal.cfg)
    assert np.array_equal(g, cal.v_codes)
    assert len(cal.v_codes) > 50_000


def test_slew_is_bounded_by_one_lsb_per_cycle(fixed_run):
    """Every step moves the register by at most one LSB, so slewing the 5 %
    error out takes at least 0.05 / 2^-v_frac cycles after the regressor
    turns on -- and sign-sign LMS gets there within a few times that."""
    cal, _ = fixed_run
    v = np.asarray(cal.v_codes)
    assert np.abs(np.diff(np.r_[cal.v_start, v])).max() == 1
    on = int(np.argmax(np.asarray(cal.r_codes) != 0))
    assert on > 0 and np.all(v[:on] == cal.v_start)   # zero regressor: no update
    target = 1 << cal.cfg.v_frac                        # gain 1.0
    n_min = cal.v_start - target                        # 0.05 * 2^16 = 3277 LSB
    crossed = int(np.argmax(v[on:] <= target))
    assert crossed > 0
    assert n_min <= crossed < 10 * n_min


def test_value_register_saturates_instead_of_wrapping():
    cfg = DpCalFixedConfig(v_bits=4, v_frac=2)
    n = 400
    e = np.where(np.arange(n) % 2, 5, -5)       # correlated +-: always +1
    up = golden_trace(e, e, cfg.v_max - 2, cfg)
    assert up.max() == cfg.v_max and up[-1] == cfg.v_max
    down = golden_trace(e, -e, 2, cfg)            # anti-correlated: always -1
    assert down.min() == 0 and down[-1] == 0


def test_exact_zero_regressor_never_updates():
    st = [1000, 0, 0]
    for e in (3, -7, 0, 12):
        assert step_codes(st, e, 0, DpCalFixedConfig()) == 1000


@pytest.mark.parametrize("cfg", [
    DpCalFixedConfig(e_bits=6, e_frac=8),
    DpCalFixedConfig(e_bits=4, e_frac=6),
    DpCalFixedConfig(reg_bits=5, reg_lsb_hz=32_000.0),
    DpCalFixedConfig(k=8),
    DpCalFixedConfig(v_bits=17, v_frac=15),
])
def test_width_sweep_still_converges(cfg):
    """Widths are configuration; sign-sign LMS only needs the signs right,
    so even a 4-bit phase error converges (measured tail 0.99675, EVM
    3.69 % at e_bits=4)."""
    cal, res = run_scenario(cfg)
    assert abs(_tail(np.asarray(cal.v_codes) / 2 ** cfg.v_frac) - 1.0) < 0.01
    assert res.evm()["evm_pct"] < 3.8


def test_value_setter_restarts_the_calibrator():
    cal = FixedSignSignLMS(init=1.2)
    cal.step(0.01, 1e5)
    cal.value = 1.05
    assert cal.code == round(1.05 * 2 ** 16) and cal.v_start == cal.code
    assert cal.e_codes == [] and abs(cal.value - 1.05) <= cal.mu / 2


def test_write_vectors_round_trip(tmp_path):
    meta = write_vectors(str(tmp_path), n_bits=200)
    e = np.loadtxt(tmp_path / "dp_cal_err.txt", dtype=np.int64)
    r = np.loadtxt(tmp_path / "dp_cal_reg.txt", dtype=np.int64)
    v = np.loadtxt(tmp_path / "dp_cal_value.txt", dtype=np.int64)
    assert e.size == r.size == v.size == meta["n"]
    assert np.array_equal(golden_trace(e, r, meta["v_start"]), v)
    dims = (tmp_path / "dp_cal_dims.svh").read_text()
    assert f"N_VEC   = {meta['n']};" in dims and f"V_START = {meta['v_start']};" in dims


def test_config_rejects_nonsense():
    with pytest.raises(ValueError):
        DpCalFixedConfig(e_bits=1)
    with pytest.raises(ValueError):
        DpCalFixedConfig(reg_lsb_hz=0.0)
    with pytest.raises(ValueError):
        golden_trace([600], [0], 0)               # 600 does not fit 10 bits


@pytest.mark.parametrize("variant", ["ble_m5pct", "sat_top", "narrow"])
def test_vector_variants(tmp_path, variant):
    """The named export variants (coverage for an RTL lockstep) are real runs: the
    files replay through golden_trace, and the calibrator still converges.
    Measured (tail / EVM): ble_m5pct 0.99836 / 3.28 %, sat_top 0.99778 / 2.67 %,
    narrow 0.99857 / 3.81 %."""
    from polartx.export.dp_cal_vectors import VARIANTS, write_variant

    meta = write_variant(str(tmp_path), variant)
    _, overrides = VARIANTS[variant]
    cfg = DpCalFixedConfig(**overrides)
    assert meta["config"] == cfg.to_dict()
    e = np.loadtxt(tmp_path / "dp_cal_err.txt", dtype=np.int64)
    r = np.loadtxt(tmp_path / "dp_cal_reg.txt", dtype=np.int64)
    v = np.loadtxt(tmp_path / "dp_cal_value.txt", dtype=np.int64)
    assert np.array_equal(golden_trace(e, r, meta["v_start"], cfg), v)
    assert abs(meta["tail_mean"] - 1.0) < 0.01
    assert meta["evm_pct"] < 4.0
    if variant == "sat_top":
        # the register really sits on its ceiling for a large part of the run
        assert meta["v_start"] == cfg.v_max and v.max() == cfg.v_max
        assert 0.1 < np.mean(v == cfg.v_max) < 0.9
    if variant == "ble_m5pct":
        # starts 5 % low and slews up by >= 90 % of the way to gain 1.0
        start, one = meta["v_start"], 1 << cfg.v_frac
        assert start < one and v[-1000:].mean() > start + 0.9 * (one - start)


def test_the_kernel_view_is_the_object_rule():
    """The engine runs the calibrator inside its per-cycle kernel
    (``fixed_dp_step`` over ``kernel_view``'s state and parameters); the
    object API steps it from Python.  Both must give the same codes for
    the same inputs -- including exact rounding ties, saturating errors and
    regressors, and a value register run into its rail."""
    from polartx.cal.dp_cal_fixed import fixed_dp_step

    cfg = DpCalFixedConfig(v_bits=12, v_frac=10)
    rng = np.random.default_rng(7)
    n = 3000
    err = rng.normal(0.0, 0.05, n)
    err[::17] = (rng.integers(-40, 40, err[::17].size) + 0.5) / 2 ** cfg.e_frac
    err[::29] = rng.choice([-1.0, 1.0], err[::29].size)       # saturates
    reg = rng.normal(0.0, 4e5, n)
    reg[::13] = (rng.integers(-40, 40, reg[::13].size) + 0.5) * cfg.reg_lsb_hz
    reg[::31] = rng.choice([-5e6, 5e6], reg[::31].size)       # saturates
    reg[: n // 2] = np.abs(reg[: n // 2]) * np.sign(err[: n // 2])  # drive V up

    obj = FixedSignSignLMS(init=3.9, cfg=cfg)
    vals_obj = [obj.step(e, r) for e, r in zip(err, reg)]

    ker = FixedSignSignLMS(init=3.9, cfg=cfg)
    step, st, pr, rec = ker.kernel_view(n)
    vals_ker = [step(st, pr, float(e), float(r), rec, i)
                for i, (e, r) in enumerate(zip(err, reg))]
    ker.kernel_done(st, rec, n)

    assert ker.e_codes == obj.e_codes and ker.r_codes == obj.r_codes
    assert ker.v_codes == obj.v_codes and ker.code == obj.code
    assert np.array_equal(vals_ker, vals_obj)
    assert max(obj.v_codes) == cfg.v_max            # the rail was reached
    assert step is fixed_dp_step
