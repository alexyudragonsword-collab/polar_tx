"""Monte Carlo yield: reproducibility and the value of calibration."""
from polartx.montecarlo import run_mc, wifi_chip_builder

N = 12   # keep CI fast; the example runs the full population


def test_reproducible_and_summary():
    b = wifi_chip_builder(bw=160e6, n_symbols=3)
    a = run_mc(b, N, limit=-35.0)
    c = run_mc(wifi_chip_builder(bw=160e6, n_symbols=3), N, limit=-35.0)
    assert (a.values == c.values).all()          # same seeds, same chips
    s = a.summary()
    assert s["n"] == N and s["std"] > 0.0
    assert 0.0 <= s["yield"] <= 1.0


def test_skew_cal_recovers_yield():
    """Per-chip skew calibration turns a skew-dominated yield loss
    around (measured on n=40: 5% -> 70%)."""
    raw = run_mc(wifi_chip_builder(bw=160e6, n_symbols=3,
                                   skew_sigma_s=0.5e-9), N, limit=-35.0)
    cal = run_mc(wifi_chip_builder(bw=160e6, n_symbols=3,
                                   skew_sigma_s=0.5e-9,
                                   calibrated_skew=True), N, limit=-35.0)
    assert cal.yield_frac > raw.yield_frac + 0.3
    assert cal.summary()["mean"] < raw.summary()["mean"] - 5.0
