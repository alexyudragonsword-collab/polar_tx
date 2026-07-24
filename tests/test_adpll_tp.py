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
