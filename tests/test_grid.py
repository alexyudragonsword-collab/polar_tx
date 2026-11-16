"""The simulation grid and the DTC / DPA update clock are separate
(polar_tx#22).  These pin what the separation must do -- leave every
default run alone, change nothing but the grid when the grid is refined --
and the physics a finer grid shows: the hold's images at the level the
sinc predicts, ACLR1 at the update instants pessimistic beyond the
pre-registered tolerance, per-tone EVM converged."""
import numpy as np
import pytest

from polartx.chain import advance
from polartx.grid import (CONVERGED_HOLD, NOT_EVIDENCE, grid_disclaimer,
                          grid_verdict)
from polartx.metrics.aclr_ext import aclr_multi
from polartx.presets import nr_dtc, wifi_dtc, wifi_outphasing
from polartx.vendor.padpd.metrics import evm_of_signal, psd

#: ex22's tolerances (array_tx's for ACLR and EVM), not tuned here
TOL = {"aclr1": 0.3, "evm_tone": 0.1}


def _wifi80(**kw):
    return wifi_dtc(bw=80e6, qam=1024, **kw)


def _run(p, noise, n_symbols=4):
    wf = p.make_waveform(n_symbols=n_symbols, seed=1)
    return p.tx.run(wf, noise=noise, seed=3)


@pytest.mark.parametrize("noise", [False, True])
def test_the_clock_on_the_grid_is_the_old_chain(noise):
    """At the defaults the clock equals the grid: the pinned preset and
    the tied one (the behaviour before the split) agree bit for bit."""
    a = _run(_wifi80(), noise)
    b = _run(_wifi80(update_oversampling=None), noise)
    assert np.array_equal(a.y, b.y)
    assert "clock_hold" not in a.info


def test_a_finer_grid_is_the_held_update_rate_output():
    """Noise off, a 4x grid on the 4x clock is the os-4 output held four
    samples per update (advanced by the hold's group delay), and the
    digital taps are the os-4 ones exactly."""
    r4 = _run(_wifi80(), False)
    r16 = _run(_wifi80(oversampling=16), False)
    assert r16.info["clock_hold"] == 4 and r16.fs == 4 * r4.fs
    assert np.array_equal(r16.env_code, r4.env_code)
    assert np.array_equal(r16.phase_out, r4.phase_out)
    ref = advance(np.repeat(r4.y, 4), 1.5)
    assert np.max(np.abs(r16.y - ref)) < 1e-12 * np.max(np.abs(ref))


def test_the_lo_below_the_clock_nyquist_is_the_update_rate_sample():
    """The grid adds LO noise above the clock's Nyquist without re-rolling
    the noise below it; jitter is one draw per update either way."""
    pm = _wifi80().tx.phasemod
    n, fs = 4096, 320e6
    j1, pn1 = pm.noise_phase(n, fs, seed=7, m=1)
    j4, pn4 = pm.noise_phase(n, fs, seed=7, m=4)
    noisy = pm.modulate(np.zeros(n), fs, noise=True, seed=7).phase_out
    clean = pm.modulate(np.zeros(n), fs, noise=False).phase_out
    assert np.allclose(noisy - clean, j1 + pn1, rtol=0, atol=1e-12)
    assert np.array_equal(j1, j4) and j4.size == n and pn4.size == 4 * n
    lo = np.fft.rfft(pn1)[:n // 2]
    assert np.allclose(np.fft.rfft(pn4)[:n // 2], 4 * lo,
                       rtol=1e-9, atol=1e-9 * np.max(np.abs(lo)))
    # and the band above holds what the PSD puts there, not zeros
    assert np.std(np.fft.rfft(pn4)[n // 2 + 1:]) > 0


def test_the_hold_puts_its_image_where_the_sinc_says():
    """On a 16x grid the first image of the 4x-clocked output sits around
    f_clock at the level the hold predicts relative to the channel -- an
    image the 4x grid folds onto the signal and cannot show.  The
    prediction is the 4-sample hold on this grid (sin(pi f m/fs) /
    (m sin(pi f/fs))); the continuous sinc it tends to is 0.9 dB lower,
    which is the image level's own grid convergence."""
    bw, m = 80e6, 4
    r = _run(_wifi80(oversampling=16), False, n_symbols=8)
    f, p = psd(r.y, r.fs, nfft=16384)
    lin = 10 ** (p / 10)
    f_clk = 4 * bw
    band = np.abs(f) <= 0.45 * bw
    image = np.abs(f - f_clk) <= 0.45 * bw
    measured = 10 * np.log10(lin[image].sum() / lin[band].sum())

    def hold(x):                      # x = f / fs, never 0 on these grids
        return np.sin(np.pi * x * m) / (m * np.sin(np.pi * x))
    g = np.linspace(-0.45 * bw, 0.45 * bw, 2000)     # even count: no f = 0
    predicted = 10 * np.log10(np.sum(hold((f_clk + g) / r.fs) ** 2)
                              / np.sum(hold(g / r.fs) ** 2))
    assert abs(measured - predicted) < 0.3, (measured, predicted)
    assert _wifi80().fs_bb / 2 < f_clk      # the os-4 grid ends below it


def _readings(p, noise):
    r = _run(p, noise)
    bw = r.wf.bw
    a = aclr_multi(r.y, r.fs, bw, offsets=(1,), nfft=1024 * int(r.fs / bw))
    ref = r.wf.require_ofdm_ref()
    return {"aclr1": max(a["aclr1_lower_dbc"], a["aclr1_upper_dbc"]),
            "evm_tone": evm_of_signal(r.y, ref, equalize="per_tone").db}


def test_the_os4_grid_reads_aclr1_pessimistic_and_per_tone_evm_right():
    """ex22's finding, on the cheapest wideband preset: with the clock
    pinned, ACLR1 at the update instants is pessimistic by more than the
    0.3 dB tolerance, os 16 is converged (os 32 within it), and per-tone
    EVM is converged already at os 4."""
    s = {k: _readings(_wifi80(oversampling=k), noise=False)
         for k in (4, 16, 32)}
    v = grid_verdict(s, TOL)
    assert v["aclr1"]["d_base"] > TOL["aclr1"]        # pessimistic, not noise
    assert v["aclr1"]["base_ok"] is False and v["aclr1"]["ref_ok"] is True
    assert v["evm_tone"]["base_ok"] is True and v["evm_tone"]["ref_ok"] is True


def test_the_verdict_is_against_the_tolerance():
    """grid_verdict on numbers chosen either side of the tolerance, and a
    reading the grid cannot hold -- None, never a pass."""
    tol = {"a": 0.3, "b": 0.1}
    s = {4: {"a": -60.0, "b": None}, 16: {"a": -60.31, "b": -40.0},
         32: {"a": -60.0, "b": -40.05}}
    v = grid_verdict(s, tol)
    assert v["a"]["base_ok"] is False            # 0.31 > 0.3
    assert v["a"]["ref_ok"] is False             # os 32 also 0.31 away
    assert v["b"]["base_ok"] is None and v["b"]["ref_ok"] is True
    s[4]["a"] = -60.2
    assert grid_verdict(s, tol)["a"]["base_ok"] is True


def test_nr_dtc_builds_its_burst_on_its_own_grid():
    """nr_dtc(oversampling=k).make_waveform() is on the preset's grid (it
    used to ignore k and always build the os-4 burst)."""
    for kw in ({"oversampling": 8}, {"oversampling": 8,
                                     "update_oversampling": None}):
        p = nr_dtc(bw=100e6, **kw)
        wf = p.make_waveform(n_symbols=2, seed=1)
        assert abs(wf.fs - p.fs_bb) < 1.0, (kw, wf.fs, p.fs_bb)
    p4, p8 = nr_dtc(bw=100e6), nr_dtc(bw=100e6, oversampling=8)
    assert p8.tx.cfg.f_clock == p4.fs_bb      # the clock stayed at 4x


def test_the_clock_must_divide_the_grid():
    with pytest.raises(ValueError, match="divide"):
        wifi_dtc(bw=80e6, oversampling=6, update_oversampling=4)


def test_outphasing_refuses_a_grid_finer_than_its_clock():
    """The split is modeled for the polar chain; outphasing ties its clock
    to the grid, and asked to hold it fails loudly instead of running half
    the chain on each rate."""
    p = wifi_outphasing(bw=80e6, oversampling=16, update_oversampling=4)
    with pytest.raises(NotImplementedError):
        _run(p, False, n_symbols=2)
    p = wifi_outphasing(bw=80e6, oversampling=8)        # tied: runs
    assert _run(p, False, n_symbols=2).fs == p.fs_bb
    # the RF-DAC chain likewise
    from polartx.cartesian import CartesianTX
    from polartx.chain import ChainConfig
    from polartx.rfdac import RFDAC, RFDACConfig
    wf = wifi_dtc(bw=80e6, oversampling=16).make_waveform(n_symbols=2, seed=1)
    ctx = CartesianTX(ChainConfig(f_clock=4 * 80e6), RFDAC(RFDACConfig()))
    with pytest.raises(NotImplementedError):
        ctx.run(wf, noise=False)


def test_the_report_says_whether_the_grid_converged():
    held = grid_disclaimer(_wifi80().tx, 320e6)
    assert held["converged"] is False and NOT_EVIDENCE in held["zh"]
    assert "采样网格未收敛" in held["zh"] and "not evidence" in held["en"]
    p16 = _wifi80(oversampling=4 * CONVERGED_HOLD)
    assert grid_disclaimer(p16.tx, p16.fs_bb)["converged"] is True
    tied = grid_disclaimer(wifi_outphasing(bw=80e6).tx, 320e6)
    assert tied["converged"] is False and "tied" in tied["en"]
    from polartx.presets import ble_1m_adpll
    p = ble_1m_adpll()
    assert grid_disclaimer(p.tx, p.fs_bb) is None


def test_the_result_reads_aclr_at_one_resolution_on_any_grid():
    """PolarResult.aclr() scales its FFT with the hold, so a finer grid is
    read at the same resolution bandwidth: on the 16x grid ACLR is the
    held output's (a few tenths of a dB better than the update instants),
    not the channel edge smeared into the adjacent band by a 4x coarser
    FFT (which read WiFi 80 MHz 4 dB worse)."""
    r4 = _run(_wifi80(), False, n_symbols=8)
    r16 = _run(_wifi80(oversampling=16), False, n_symbols=8)
    a4, a16 = r4.aclr()["upper_dbc"], r16.aclr()["upper_dbc"]
    assert r16.grid_hold == 4 and r4.grid_hold == 1
    assert 0.0 < a4 - a16 < 2.0, (a4, a16)

