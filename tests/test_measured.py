"""Measured-data DPA modeling: synthetic round-trip (always) and the
real OpenDPD captures (skipped when the dataset clone is absent)."""
import numpy as np
import pytest

from polartx.dpa import DPA, DPAConfig
from polartx.measured import (dpa_from_measured, extract_polar_characteristics,
                              find_opendpd_root, load_measured_dpa)


def test_synthetic_roundtrip():
    """Feed a known Rapp+AM-PM device; the extraction recovers it."""
    rng = np.random.default_rng(3)
    n = 1 << 16
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.25
    x = np.clip(np.abs(x), 0, 1.0) * np.exp(1j * np.angle(x))
    ref = DPA(DPAConfig(n_bits=12, amam=("rapp", 2.5, 1.2),
                        ampm_deg_poly=(0.0, 3.0, 4.0)))
    y = 2.0 * ref(ref.encode(np.abs(x)), np.angle(x))
    ch = extract_polar_characteristics(x, y)
    # static device -> static fit (floor: 64-bin LUT granularity ~ -33 dB,
    # far below the real DPA's memory-limited -20 dB)
    assert ch["static_nmse_db"] < -30.0
    # median |y/x|: 2.0 scale x the Rapp small-signal slope (~1.56) since
    # most samples sit well below compression
    assert 2.0 < ch["gain"] < 3.6
    # AM-PM shape recovered within 0.5 deg over the upper half
    hi = ch["r_in"] > 0.5
    expect = 3.0 * ch["r_in"][hi] + 4.0 * ch["r_in"][hi] ** 2
    got = ch["ampm_deg"][hi] - ch["ampm_deg"][hi][0] + expect[0]
    assert np.max(np.abs(got - expect)) < 0.5


def test_dpa_from_measured_builds():
    rng = np.random.default_rng(4)
    n = 1 << 14
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.3
    y = 1.5 * x * np.exp(1j * 0.05 * np.abs(x))
    dpa, ch = dpa_from_measured(x, y)
    assert dpa.amp_table.size == 1 << 10
    assert np.all(np.diff(dpa.amp_table) >= -1e-9)


needs_data = pytest.mark.skipif(find_opendpd_root() is None,
                                reason="OpenDPD dataset clone not found")


@needs_data
def test_real_dpa_160mhz():
    """The real OpenDPD DPA: memory-dominated (static-polar NMSE ~ -20 dB
    vs GMP-510's published -39 dB) — the number that motivates the
    Cartesian memory DPD."""
    dpa, ch = load_measured_dpa("DPA_160MHz")
    assert -24.0 < ch["static_nmse_db"] < -16.0
    assert abs(ch["align"]["lag_total"]) < 0.5   # capture pre-aligned
    span = ch["ampm_deg"].max() - ch["ampm_deg"].min()
    assert 4.0 < span < 15.0                     # measured: 8.5 deg


@needs_data
def test_real_dpa_chain_with_polar_dpd():
    from polartx.cal.polar_dpd import PolarDPD
    from polartx.chain import ChainConfig, PolarTX
    from polartx.phasemod import IdealPhaseModulator
    from polartx.waveforms.ofdm import wifi_waveform
    dpa, _ = load_measured_dpa("DPA_160MHz")
    wf = wifi_waveform(160e6, 1024, n_symbols=4, seed=1)
    r0 = PolarTX(ChainConfig(env_floor=0.02), IdealPhaseModulator(),
                 dpa).run(wf, noise=False)
    r1 = PolarTX(ChainConfig(env_floor=0.02), IdealPhaseModulator(),
                 dpa, dpd=PolarDPD.from_dpa(dpa)).run(wf, noise=False)
    assert r1.evm().db < r0.evm().db - 12.0      # measured: -32 -> -50
