"""Literature-class benchmark presets stay in their published classes."""
from polartx.metrics.aclr_ext import aclr_multi
from polartx.presets import (bench_edge_polar_staszewski05,
                             bench_lte20_polar_madoglio14,
                             bench_wifi11n_polar)


def test_edge_polar_class():
    p = bench_edge_polar_staszewski05()
    r = p.tx.run(p.make_waveform(400, seed=1), noise=True, seed=1)
    d = r.evm()["devm_pct"]
    assert 1.0 < d < 3.5           # published class 2-3%, spec 9%


def test_lte20_polar_class():
    p = bench_lte20_polar_madoglio14()
    r = p.tx.run(p.make_waveform(n_symbols=8, seed=0), noise=True, seed=1)
    assert -34.0 < r.evm().db < -28.0        # class ~-30 dB
    a = aclr_multi(r.y, r.fs, 20e6, offsets=(1,))
    assert max(a["aclr1_lower_dbc"], a["aclr1_upper_dbc"]) < -33.0


def test_wifi11n_polar_class():
    p = bench_wifi11n_polar()
    r = p.tx.run(p.make_waveform(n_symbols=6, seed=0), noise=True, seed=1)
    assert -32.0 < r.evm().db < -25.0        # class ~-28, spec -25
