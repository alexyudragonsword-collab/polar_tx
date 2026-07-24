"""Spec-based parallel Monte Carlo: determinism and worker equivalence."""
import numpy as np

from polartx.montecarlo import run_mc_parallel

SPEC = {"bw": 160e6, "n_symbols": 2, "sigma_cell": 0.008,
        "dtc_gain_sigma": 0.01, "inl_sigma_ui": 1e-3, "lo_sigma_db": 1.0,
        "skew_sigma_s": 0.3e-9}


def test_serial_deterministic():
    a = run_mc_parallel(SPEC, 6)
    b = run_mc_parallel(SPEC, 6)
    assert np.array_equal(a.values, b.values)
    assert a.summary()["std"] > 0.0


def test_parallel_matches_serial():
    a = run_mc_parallel(SPEC, 6)
    b = run_mc_parallel(SPEC, 6, n_workers=2)
    assert np.allclose(a.values, b.values)


def test_expanded_draws_widen_spread():
    tight = run_mc_parallel({"bw": 160e6, "n_symbols": 2,
                             "sigma_cell": 0.001, "dtc_gain_sigma": 0.0,
                             "lo_sigma_db": 0.0}, 8)
    wide = run_mc_parallel(SPEC, 8)
    assert wide.summary()["std"] > 2.0 * max(tight.summary()["std"], 1e-3)
