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
