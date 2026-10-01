"""Vendor subtree sanity: everything imports and the engines still work."""
import numpy as np


def test_imports():
    import polartx.vendor.pllsim.arch.adpll  # noqa: F401
    import polartx.vendor.pllsim.arch.frac  # noqa: F401
    import polartx.vendor.pllsim.blocks.dtc  # noqa: F401
    import polartx.vendor.pllsim.calibration.lms  # noqa: F401
    import polartx.vendor.pllsim.core.dtcspurs  # noqa: F401
    import polartx.vendor.pllsim.modulation  # noqa: F401
    import polartx.vendor.pllsim.synth  # noqa: F401
    import polartx.vendor.padpd.cfr  # noqa: F401
    import polartx.vendor.padpd.data.align  # noqa: F401
    import polartx.vendor.padpd.deploy.fixed_point  # noqa: F401
    import polartx.vendor.padpd.metrics  # noqa: F401
    import polartx.vendor.padpd.pa  # noqa: F401
    import polartx.vendor.padpd.waveform  # noqa: F401


def _adpll(fref=100e6, fout=10e9):
    from polartx.vendor.pllsim.arch.adpll import ADPLL, ADPLLConfig, DLFConfig
    from polartx.vendor.pllsim.blocks.oscillator import OscConfig
    from polartx.vendor.pllsim.blocks.tdc import TDCConfig
    from polartx.vendor.pllsim.synth import design_adpll_dlf

    alpha, rho = design_adpll_dlf(fref, 1e6, 55.0)
    osc = OscConfig(f0=fout, gain=30e3, pn_dbchz=-110.0, pn_foffset=1e6,
                    pn_f1f3=300e3, pn_floor_dbchz=-150.0)
    cfg = ADPLLConfig(fref=fref, fout=fout, osc=osc,
                      dlf=DLFConfig(alpha=alpha, rho=rho), mode="tdc",
                      tdc=TDCConfig(t_res=1e-12))
    return ADPLL(cfg)


def test_adpll_analyze_and_simulate():
    pll = _adpll()
    res = pll.analyze()
    assert 10.0 < res.jitter_fs < 1000.0
    sim = pll.simulate(20000, seed=1)
    assert sim.lock_time_s is not None
    assert np.isfinite(sim.phase_err_out[-1000:]).all()


def test_ofdm_qam_evm_loopback():
    from polartx.vendor.padpd.metrics import evm_of_signal
    from polartx.vendor.padpd.waveform import OFDMConfig, generate_ofdm

    wf = generate_ofdm(OFDMConfig(bandwidth_hz=20e6, qam_order=256,
                                  n_symbols=4, seed=3))
    r = evm_of_signal(wf.x, wf, equalize="scalar")
    assert r.db < -80.0


def test_load_model_reads_an_npz_written_by_upstream_padpd():
    """Cross-library round trip.  tests/data/padpd_gmp_roundtrip.npz was
    written by PA_DPD's own ``GMPModel.save`` (commit recorded in the
    companion _io.npz, 44cbcb3 at creation); the vendored ``load_model``
    must rebuild the same class with the same config and reproduce the
    output upstream computed on the stored input — same basis, same
    coefficients.  To floating-point precision, not bit for bit: the
    basis @ coeffs product goes through BLAS, whose summation order
    differs between platforms (the bit-exact form passed on x86 Linux
    and failed on macOS arm64 and the dependency-floor job)."""
    import os
    import numpy as np
    from polartx.vendor.padpd.pa import GMPModel, load_model
    here = os.path.join(os.path.dirname(__file__), "data")
    model = load_model(os.path.join(here, "padpd_gmp_roundtrip.npz"))
    io = np.load(os.path.join(here, "padpd_gmp_roundtrip_io.npz"))
    assert isinstance(model, GMPModel)
    assert model.get_config() == {"order": 3, "memory_depth": 2,
                                  "lag_order": 2, "lag_memory": 1, "lag_span": 1,
                                  "lead_order": 2, "lead_memory": 1, "lead_span": 1}
    assert model.coeffs.size == int(io["n_coeffs"]) == model.n_coeffs
    y = model(io["x"])
    assert np.allclose(y, io["y_model"], rtol=1e-12, atol=1e-14)
    assert np.max(np.abs(y - io["y_model"])) < 1e-13 * np.max(np.abs(y))
    assert str(io["padpd_commit"]) == "44cbcb3"
