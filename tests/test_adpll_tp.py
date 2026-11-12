"""ADPLL two-point phase modulator: mode cross-check and mismatch laws."""
import numpy as np

from polartx.presets import ble_1m_adpll


def _evm_db(preset, wf, **run_kw):
    return preset.tx.run(wf, **run_kw).evm()["evm_db"]


def test_response_vs_event_evm_agree():
    """The fast linearized model and the cycle-accurate engine must tell
    the same noisy-EVM story (32 samples/symbol, matched gains)."""
    pr = ble_1m_adpll(mode="response")
    pe = ble_1m_adpll(mode="event")
    wf = pr.make_waveform(n_bits=600, seed=5)
    e_r = _evm_db(pr, wf, noise=True, seed=3)
    e_e = _evm_db(pe, wf, noise=True, seed=3)
    assert abs(e_r - e_e) < 1.5


def test_dp_gain_error_law():
    """The direct-path gain-error residual is exactly linear in eps at
    the modulator output: phase_out - phase_cmd = eps * highpass(phase).
    (The EVM estimator's lag/detrend recovery absorbs part of it, so the
    strict 6 dB/octave is asserted on the raw residual.)"""
    wf = ble_1m_adpll().make_waveform(n_bits=600, seed=5)
    _, phase = np.abs(wf.x), np.unwrap(np.angle(wf.x))
    rms = {}
    for eps in (0.02, 0.04, 0.08):
        pm = ble_1m_adpll(dp_gain=1.0 + eps).tx.phasemod
        out = pm.modulate(phase, 32e6, noise=False).phase_out
        rms[eps] = float(np.std(out - phase))
    assert abs(rms[0.04] / rms[0.02] - 2.0) < 0.02
    assert abs(rms[0.08] / rms[0.04] - 2.0) < 0.02
    # and end-to-end EVM still degrades monotonically with eps
    e = {eps: _evm_db(ble_1m_adpll(dp_gain=1.0 + eps), wf, noise=False)
         for eps in (0.02, 0.08)}
    assert e[0.08] > e[0.02] + 3.0


def test_matched_evm_independent_of_loop_bw():
    """The whole point of two-point modulation: with matched gains the
    loop bandwidth does not touch the modulation (noiseless residual only;
    noisy EVM differs only through the noise budget)."""
    wf = ble_1m_adpll().make_waveform(n_bits=600, seed=5)
    e = [_evm_db(ble_1m_adpll(loop_bw=bw), wf, noise=False)
         for bw in (50e3, 150e3, 400e3)]
    assert max(e) < -80.0          # matched + noiseless = numerically exact


def test_response_vs_event_psd_band_average():
    """pllsim discipline at the spectrum level: the linearized response
    engine and the cycle-accurate event engine must agree band-averaged
    within 3 dB over the modulation band, not just on scalar EVM."""
    import numpy as np

    from polartx.vendor.padpd.metrics import psd

    wf = ble_1m_adpll().make_waveform(n_bits=1500, seed=5)
    ys = {}
    for mode in ("response", "event"):
        p = ble_1m_adpll(mode=mode)
        ys[mode] = p.tx.run(wf, noise=True, seed=3).y
    f_r, p_r = psd(ys["response"], 32e6, nfft=4096)
    f_e, p_e = psd(ys["event"], 32e6, nfft=4096)
    band = (np.abs(f_r) > 0.05e6) & (np.abs(f_r) < 8e6)
    # band-average in ~octave bins to avoid bin-by-bin noise
    edges = np.geomspace(0.05e6, 8e6, 10)
    diffs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = band & (np.abs(f_r) >= lo) & (np.abs(f_r) < hi)
        if m.sum() >= 4:
            diffs.append(abs(p_r[m].mean() - p_e[m].mean()))
    assert max(diffs) < 3.0


def test_fractional_ble_channels():
    """Real BLE hops 2.402-2.480 GHz: fractional FCW channels give the
    same EVM as the integer-friendly 2.44 GHz preset (TDC-mode ADPLL
    supports fractional FCW natively)."""
    e = {}
    for fout in (2.402e9, 2.440e9, 2.479e9):
        p = ble_1m_adpll(fout=fout)
        wf = p.make_waveform(n_bits=400, seed=5)
        e[fout] = p.tx.run(wf, noise=True, seed=3).evm()["evm_db"]
    assert max(e.values()) - min(e.values()) < 1.0


def test_kdco_error_acts_as_dp_gain_error():
    """A Kdco estimate error scales the direct point by 1/(1+err) — same
    residual law as an explicit dp-gain error of the same size."""
    wf = ble_1m_adpll().make_waveform(n_bits=600, seed=5)
    e_k = _evm_db(ble_1m_adpll(kdco_est_error=0.05), wf, noise=False)
    e_g = _evm_db(ble_1m_adpll(dp_gain=1.0 / 1.05), wf, noise=False)
    assert abs(e_k - e_g) < 0.5


def _phase_err_rms(r):
    """RMS of phase_out - phase_cmd after removing static phase and
    frequency offset (what a receiver's carrier recovery takes out)."""
    e = r.phase_out - r.phase_cmd
    k = np.arange(e.size)
    a, b = np.polyfit(k, e, 1)
    return float(np.std(e - (a * k + b)))


def test_the_tdc_must_span_a_dco_period():
    """The counter-assisted loop reads the fraction of a DCO cycle with the
    TDC; a shorter range saturates and the loop is blind for the rest of
    the cycle.  The LTE-20 preset shipped that way (1 ps x 7 bit = 127 ps
    against 513 ps) and its event-engine EVM of -15 dB was read as an
    engine limit for three months (ROADMAP B1).  Now it cannot be built."""
    import dataclasses

    import pytest

    from polartx.phasemod.adpll_tp import ADPLLTwoPoint
    from polartx.presets import lte20_adpll
    from polartx.vendor.pllsim.arch.adpll import ADPLL

    pll = lte20_adpll().tx.phasemod.pll
    c = pll.cfg
    span = ((1 << c.tdc.n_bits) - 1) * c.tdc.t_res
    assert span >= 1.0 / c.fout
    short = dataclasses.replace(c, tdc=dataclasses.replace(c.tdc, n_bits=7))
    with pytest.raises(ValueError, match="saturates every cycle"):
        ADPLLTwoPoint(ADPLL(short))


def test_event_engine_has_no_phase_advance_limit():
    """The event engine does not need the phase to advance by much less
    than one UI per reference cycle: the counter takes whole DCO cycles and
    the FCW path carries the modulation, so the TDC only sees the residual.

    LTE-20 advances ~30% of a UI per cycle.  With the TDC spanning a period
    its error is bounded by the one term the linear model lacks, the TDC's
    normalization by the NOMINAL period, which misreads a fraction of a
    cycle by at most df/f; so the output error cannot exceed
    2*pi*rms(df)/f.  (With the old 7-bit TDC it was 8x over that bound.)
    Where df/f is ~1e-4, BLE and EDR, nothing separates the engines.
    """
    import warnings

    from polartx.presets import bt_edr_adpll, lte20_adpll

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        t = lte20_adpll(mode="event", dpd=False)
        rt = t.tx.run(t.make_waveform(n_symbols=4, seed=0), noise=False,
                      seed=1)
    diag = rt.info["phasemod"]
    assert diag["ui_per_ref_cycle_p99"] > 0.20          # far past the old rule
    f_dev = np.diff(rt.phase_cmd) * rt.fs / (2 * np.pi)
    ceiling = (2 * np.pi * np.sqrt(np.mean(f_dev ** 2))
               / t.tx.phasemod.pll.cfg.fout)
    assert _phase_err_rms(rt) < ceiling

    for preset, wkw in ((ble_1m_adpll, dict(n_bits=400, seed=5)),
                        (lambda **k: bt_edr_adpll("8dpsk", **k),
                         dict(n_syms=300, seed=1))):
        p = preset(mode="event")
        wf = p.make_waveform(**wkw)
        r_ev = p.tx.run(wf, noise=True, seed=3)
        assert r_ev.info["phasemod"]["dco_dev_frac_p99"] < 1e-3
        e_ev = _phase_err_rms(r_ev)
        p.tx.phasemod.mode = "response"
        e_rs = _phase_err_rms(p.tx.run(wf, noise=True, seed=3))
        assert abs(20 * np.log10(e_ev / e_rs)) < 1.0
