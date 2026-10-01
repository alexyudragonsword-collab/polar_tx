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


# ------------------------------------------------ residual memory model
def _synthetic_dut(n_symbols=16, seed=1):
    """Rapp + AM-PM static device followed by a known 2-tap FIR and a cubic
    term: the memory the residual model has to recover.  Band-limited
    (OFDM at 4x oversampling) so the device's sub-sample group delay is
    a legitimate linear-memory feature, not a resampling artefact."""
    from polartx.waveforms.ofdm import wifi_waveform
    wf = wifi_waveform(80e6, 256, n_symbols=n_symbols, seed=seed)
    x = wf.x / np.abs(wf.x).max()
    ref = DPA(DPAConfig(n_bits=14, amam=("rapp", 2.5, 1.2),
                        ampm_deg_poly=(0.0, 3.0, 4.0)))
    s = 2.0 * ref(ref.encode(np.abs(x)), np.angle(x))
    fir = np.array([1.0, 0.12 + 0.04j])
    lin = np.convolve(s, fir, mode="full")[:s.size]
    return wf, x, lin + 0.05 * lin * np.abs(lin) ** 2, fir


def test_static_prediction_is_the_nmse_code_path():
    rng = np.random.default_rng(5)
    n = 1 << 14
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.3
    y = 1.5 * x * np.exp(1j * 0.05 * np.abs(x))
    ch = extract_polar_characteristics(x, y)
    from polartx.measured import static_prediction
    from polartx.vendor.padpd.pa import nmse_db
    assert nmse_db(ch["y_aligned"], static_prediction(ch, ch["x_aligned"])) == ch["static_nmse_db"]
    assert ch["fs_scale"] == np.abs(ch["x_aligned"]).max()
    assert ch["align"]["fractional_removed"] is False


def test_synthetic_dut_residual_recovers_the_injected_memory():
    """Process correctness, no data needed.  Static extraction + residual
    GMP on a device with a KNOWN memory: the composite must explain the
    output to < -50 dB and the residual's linear taps must be the injected
    FIR (relative tap) within 5 %.  Integer-only alignment is what makes
    this possible: with the sub-sample group delay resampled away the
    composite stalls at -46.6 dB (measured) because a short FIR cannot
    undo a sinc interpolator."""
    from polartx.measured import fit_residual_memory, linear_taps
    from polartx.vendor.padpd.pa import GMPModel
    wf, x, y, fir = _synthetic_dut()
    dpa, ch = dpa_from_measured(x, y, n_bits=12)
    assert abs(ch["align"]["lag_total"] - 0.10) < 0.05  # the FIR's group delay, left in
    assert ch["y_aligned"].size == y.size                 # nothing resampled
    model, rep = fit_residual_memory(
        ch, lambda: GMPModel(order=5, memory_depth=3, lag_order=0, lag_memory=0,
                             lag_span=0, lead_order=0, lead_memory=0, lead_span=0),
        regularization=1e-12)
    assert rep["residual_nmse_db"] < -50.0, rep           # measured -52.8
    assert rep["static_nmse_db"] > -30.0                  # the memory is visible
    taps = linear_taps(model)
    rel = taps[1] / taps[0]
    assert abs(rel - fir[1]) / abs(fir[1]) < 0.05, (rel, fir[1])   # measured 2.3 %
    assert abs(taps[2]) < 0.02 * abs(taps[0])             # no third tap was injected


def test_residual_model_round_trips_through_npz_bit_exactly(tmp_path):
    """Save with PAModel.save, read with the vendored load_model, put both
    in the chain: identical output.  The chain is built the way the
    docstring demands -- fs_scale_fixed at the capture's full scale, the
    waveform peaking there, same sample rate."""
    from dataclasses import replace
    from polartx.chain import ChainConfig, PolarTX
    from polartx.measured import fit_residual_memory
    from polartx.phasemod import IdealPhaseModulator
    from polartx.vendor.padpd.pa import GMPModel, load_model
    wf, x, y, _ = _synthetic_dut(n_symbols=8)
    dpa, ch = dpa_from_measured(x, y, n_bits=12)
    model, _ = fit_residual_memory(
        ch, lambda: GMPModel(order=3, memory_depth=2, lag_order=0, lag_memory=0,
                             lag_span=0, lead_order=0, lead_memory=0, lead_span=0))
    path = tmp_path / "residual.npz"
    model.save(str(path))
    back = load_model(str(path))
    cfg = ChainConfig(fs_scale_fixed=ch["fs_scale"])
    drive = replace(wf, x=wf.x / np.abs(wf.x).max() * ch["fs_scale"])
    y1 = PolarTX(cfg, IdealPhaseModulator(), dpa, memory=model).run(drive, noise=False).y
    y2 = PolarTX(cfg, IdealPhaseModulator(), dpa, memory=back).run(drive, noise=False).y
    assert np.array_equal(y1, y2)
    assert type(back) is type(model) and back.get_config() == model.get_config()


@pytest.fixture(scope="module")
def dpa160():
    pytest.importorskip("numpy")
    if find_opendpd_root() is None:
        pytest.skip("OpenDPD dataset clone not found")
    return load_measured_dpa("DPA_160MHz", with_memory=True)


@needs_data
def test_real_dpa_160mhz_residual_memory(dpa160):
    """Acceptance on the real device.  Validation-set numbers, chain scale.
    GMP-510 residual on top of the 64-bin LUT must reach -36 dB (OpenDPD's
    direct GMP-510 fit is -39.2 published, -38.7 here with the same
    split); the memoryless control must buy ~nothing, which is what makes
    '19 dB is memory' a fair statement; and the chain built as documented
    must reproduce the measured output to the model's own accuracy."""
    from dataclasses import replace
    from polartx.chain import ChainConfig, PolarTX
    from polartx.measured import residual_training_pair
    from polartx.phasemod import IdealPhaseModulator
    from polartx.vendor.padpd.pa import GMPModel, nmse_db
    from polartx.waveforms.base import Waveform
    dpa, ch = dpa160
    rep = ch["memory_report"]
    assert isinstance(ch["memory"], GMPModel) and rep["n_coeffs"] == 255
    assert -24.0 < rep["static_nmse_db"] < -16.0               # -19.95
    assert rep["static_nmse_db"] - rep["memoryless_nmse_db"] < 0.5   # measured 0.02 dB
    assert rep["residual_nmse_db"] <= -36.0, rep               # measured -37.5
    assert abs(rep["residual_train_nmse_db"] - rep["residual_nmse_db"]) < 1.0  # no over-fit
    # the chain, driven with the captured input, reproduces the capture
    u, v = residual_training_pair(ch)
    wf = Waveform(x=ch["x_aligned"], fs=ch["fs"], bw=160e6, kind="ofdm")
    cfg = ChainConfig(fs_scale_fixed=ch["fs_scale"])
    val = slice(rep["n_train"], None)
    y0 = PolarTX(cfg, IdealPhaseModulator(), dpa).run(wf, noise=False).y
    y1 = PolarTX(cfg, IdealPhaseModulator(), dpa, memory=ch["memory"]).run(wf, noise=False).y
    assert nmse_db(u[val], y0[val]) < -50.0                    # static path == u (10-bit quant)
    assert nmse_db(v[val], y1[val]) <= -35.0                   # measured -37.8
    assert abs(nmse_db(v[val], y1[val]) - rep["residual_nmse_db"]) < 1.0
    _ = replace  # noqa: F841 - keep the import grouped with the chain ones


@needs_data
def test_real_dpa_200mhz_residual_memory():
    dpa, ch = load_measured_dpa("DPA_200MHz", with_memory=True)
    rep = ch["memory_report"]
    assert rep["residual_nmse_db"] <= -31.0, rep               # measured -33.3 (published -33.7)
    assert rep["static_nmse_db"] - rep["memoryless_nmse_db"] < 0.5   # 0.02 dB


@needs_data
def test_spline_gmp_residual_matches_gmp_with_a_better_conditioned_basis(dpa160):
    """Same 255-coefficient budget: SplineGMP within 0.5 dB of GMP-510 and
    a condition number at least an order of magnitude lower.  The design
    note expected 2-3 orders (PA_DPD's direct x->y fits); on the residual
    problem the input is an already-compressed envelope and the
    polynomial basis is far less collinear, so the gap is ~1.4 orders
    (2.1e5 vs 8.8e3) -- asserted at what is measured, recorded in
    cairn/measured-memory.md."""
    from polartx.measured import fit_residual_memory, residual_training_pair, spline_gmp_residual
    dpa, ch = dpa160
    rep = ch["memory_report"]
    u, _ = residual_training_pair(ch)
    model_s, rep_s = fit_residual_memory(ch, lambda: spline_gmp_residual(u[:rep["n_train"]]))
    assert rep_s["n_coeffs"] == rep["n_coeffs"] == 255
    assert abs(rep_s["residual_nmse_db"] - rep["residual_nmse_db"]) <= 0.5   # -37.7 vs -37.5
    assert rep["condition_number"] / rep_s["condition_number"] >= 10.0       # 24x


@needs_data
def test_memory_in_chain_starves_the_polar_dpd(dpa160):
    """Acceptance: with the residual memory in the chain the polar LUT DPD,
    which fixes AM-AM/AM-PM only, buys almost nothing -- the proof that
    the memory really reaches the output and is not code-static."""
    from dataclasses import replace
    from polartx.cal.polar_dpd import PolarDPD
    from polartx.chain import ChainConfig, PolarTX
    from polartx.phasemod import IdealPhaseModulator
    from polartx.waveforms.ofdm import wifi_waveform
    dpa, ch = dpa160
    wf = wifi_waveform(160e6, 1024, n_symbols=4, seed=1)
    wf = replace(wf, x=wf.x / np.abs(wf.x).max() * ch["fs_scale"])
    cfg = ChainConfig(env_floor=0.02, fs_scale_fixed=ch["fs_scale"])
    dpd = PolarDPD.from_dpa(dpa)
    e = {(d, m): PolarTX(cfg, IdealPhaseModulator(), dpa, dpd=dpd if d else None,
                         memory=ch["memory"] if m else None).run(wf, noise=False).evm().db
         for d in (0, 1) for m in (0, 1)}
    gain_static = e[0, 0] - e[1, 0]
    gain_memory = e[0, 1] - e[1, 1]
    assert gain_static > 12.0                 # measured -32.0 -> -49.9
    assert gain_memory < 3.0                  # measured -19.75 -> -19.94
    assert e[0, 1] > e[0, 0] + 8.0            # the memory costs > 8 dB of EVM (12.3)
