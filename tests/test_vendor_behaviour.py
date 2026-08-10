"""Vendored primitives verified against analytic ground truth.

The vendor subtree is 37 byte-identical files carried from the sibling
repos, and the drift checker proves only that they have not CHANGED — not
that they are correct. Every metric in this library reduces to these
primitives, so each one is pinned here against a value derivable by hand,
independently of the implementation.
"""
import numpy as np
import pytest

from polartx.vendor.padpd.cfr import cfr_clip_filter
from polartx.vendor.padpd.data.align import align_delay
from polartx.vendor.padpd.deploy.fixed_point import quantize_symmetric
from polartx.vendor.padpd.metrics.aclr import aclr
from polartx.vendor.padpd.metrics.ccdf import ccdf
from polartx.vendor.padpd.metrics.evm import evm
from polartx.vendor.padpd.metrics.spectrum import psd
from polartx.vendor.padpd.waveform.qam import (qam_constellation, qam_demodulate,
                                               qam_modulate)


# --------------------------------------------------------------- EVM
def test_evm_equals_the_injected_noise_ratio():
    """EVM is by definition the error-to-reference amplitude ratio, so a
    known AWGN level must read back as exactly that level."""
    rng = np.random.default_rng(0)
    tx = qam_modulate(rng.integers(0, 64, (8, 512)), 64)
    for snr_db in (20.0, 30.0, 40.0):
        sig = np.sqrt(np.mean(np.abs(tx) ** 2))
        n = sig * 10 ** (-snr_db / 20) / np.sqrt(2) * (
            rng.standard_normal(tx.shape) + 1j * rng.standard_normal(tx.shape))
        got = evm(tx + n, tx).db
        assert got == pytest.approx(-snr_db, abs=0.5)


def test_scalar_equalization_removes_a_common_gain_and_rotation():
    rng = np.random.default_rng(1)
    tx = qam_modulate(rng.integers(0, 16, (4, 256)), 16)
    rx = tx * (0.63 * np.exp(1j * 0.9))          # arbitrary gain + rotation
    assert evm(rx, tx, equalize="scalar").db < -100


def test_per_tone_equalization_removes_a_frequency_response():
    """A per-subcarrier gain is exactly what a receiver's equalizer takes
    out — scalar equalization must still see it."""
    rng = np.random.default_rng(2)
    tx = qam_modulate(rng.integers(0, 16, (16, 128)), 16)
    h = (0.5 + rng.random(128)) * np.exp(1j * rng.uniform(-1, 1, 128))
    rx = tx * h                                   # same channel every symbol
    assert evm(rx, tx, equalize="per_tone").db < -100
    assert evm(rx, tx, equalize="scalar").db > -15


def test_per_tone_needs_a_two_dimensional_grid():
    with pytest.raises(ValueError):
        evm(np.ones(8), np.ones(8), equalize="per_tone")


# --------------------------------------------------------------- QAM
@pytest.mark.parametrize("order", [4, 16, 64, 256, 1024])
def test_qam_constellation_is_unit_power_and_square(order):
    c = qam_constellation(order)
    assert c.size == order
    assert np.mean(np.abs(c) ** 2) == pytest.approx(1.0, rel=1e-9)
    n_lvl = len(np.unique(np.round(c.real, 9)))
    assert n_lvl == int(round(np.sqrt(order)))    # square QAM


@pytest.mark.parametrize("order", [4, 16, 64, 256])
def test_qam_round_trips(order):
    rng = np.random.default_rng(3)
    sym = rng.integers(0, order, 4000)
    assert np.array_equal(qam_demodulate(qam_modulate(sym, order), order), sym)


# --------------------------------------------------------------- PSD
def test_psd_locates_a_tone_and_conserves_power():
    fs, n = 1e6, 1 << 14
    t = np.arange(n) / fs
    x = 1.7 * np.exp(2j * np.pi * 1.25e5 * t)
    f, p = psd(x, fs, nfft=4096)
    assert f[np.argmax(p)] == pytest.approx(1.25e5, abs=fs / 4096)
    assert p.max() == pytest.approx(0.0, abs=1e-6)      # peak-normalized dB


# --------------------------------------------------------------- ACLR
def test_aclr_reports_a_known_adjacent_channel_level():
    """Wanted carrier plus one adjacent-channel tone at a known ratio."""
    fs, n, bw = 100e6, 1 << 16, 20e6
    t = np.arange(n) / fs
    for ratio_db in (-30.0, -45.0):
        x = (np.exp(2j * np.pi * 1e6 * t)
             + 10 ** (ratio_db / 20) * np.exp(2j * np.pi * 21e6 * t))
        got = aclr(x, fs, bw, nfft=4096)["upper_dbc"]
        assert got == pytest.approx(ratio_db, abs=2.0)


# --------------------------------------------------------------- CFR
def test_cfr_reduces_papr_toward_target_and_keeps_average_power():
    rng = np.random.default_rng(4)
    fs, bw, n = 200e6, 40e6, 1 << 15
    spec = np.zeros(n, dtype=complex)
    k = int(bw / fs * n) // 2
    spec[1:k] = rng.standard_normal(k - 1) + 1j * rng.standard_normal(k - 1)
    spec[-k:] = rng.standard_normal(k) + 1j * rng.standard_normal(k)
    x = np.fft.ifft(spec)

    def papr_db(v):
        a = np.abs(v) ** 2
        return 10 * np.log10(a.max() / a.mean())

    p0 = papr_db(x)
    y = cfr_clip_filter(x, 7.0, fs, bw)
    assert papr_db(y) < p0 - 1.0                    # it does reduce PAPR
    assert papr_db(y) < 9.0                          # ...toward the target
    rms = lambda v: np.sqrt(np.mean(np.abs(v) ** 2))  # noqa: E731
    assert rms(y) == pytest.approx(rms(x), rel=0.15)  # average power kept


def test_cfr_is_a_no_op_below_the_target():
    rng = np.random.default_rng(5)
    x = np.exp(2j * np.pi * rng.random(4096))        # constant envelope
    y = cfr_clip_filter(x, 12.0, 100e6, 20e6)
    assert np.allclose(x, y)


# --------------------------------------------------------------- align
@pytest.mark.parametrize("lag,gain", [(7, 1.0), (-5, 0.4),
                                      (13, 2.2), (0, 1.0)])
def test_align_delay_recovers_a_known_lag_and_gain(lag, gain):
    rng = np.random.default_rng(6)
    x = rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
    y = gain * np.roll(x, lag)
    xa, ya, info = align_delay(x, y)
    assert abs(info["lag_total"] - lag) < 0.05
    assert abs(info["gain"]) == pytest.approx(gain, rel=0.05)
    # after alignment the two agree
    m = slice(64, -64)
    assert np.allclose(ya[m] / info["gain"], xa[m], atol=1e-6)


# --------------------------------------------------- fixed point / CCDF
@pytest.mark.parametrize("n_bits", [6, 8, 12, 14])
def test_quantize_symmetric_hits_the_expected_lsb(n_bits):
    """The documented contract is a POWER-OF-TWO step chosen from the array
    peak — not a full-scale/(2^(n-1)-1) LSB. Pin the step it implies, and
    the uniform-quantization error law that follows from it."""
    rng = np.random.default_rng(7)
    x = rng.uniform(-1, 1, 20000)
    q = quantize_symmetric(x, n_bits)
    # documented step: the power of two that lets 2^(n-1)-1 codes cover the
    # array peak (so it is rounded UP, and the format has headroom)
    step = 2.0 ** np.ceil(np.log2(np.abs(x).max() / (2 ** (n_bits - 1) - 1)))
    err = q - x
    assert np.max(np.abs(err)) <= step / 2 * 1.001        # pure rounding
    assert np.std(err) == pytest.approx(step / np.sqrt(12), rel=0.05)


def test_ccdf_of_a_constant_envelope_collapses_to_zero_papr():
    """A constant envelope has no peaks above its mean power, so the CCDF
    spans essentially no dB at all."""
    x = np.exp(2j * np.pi * np.linspace(0, 10, 4096))
    db, _ = ccdf(x)
    assert db.max() < 0.01                     # PAPR ~ 0 dB

    # ...whereas a Gaussian-ish OFDM envelope reaches several dB
    rng = np.random.default_rng(8)
    g = (rng.standard_normal(1 << 14) + 1j * rng.standard_normal(1 << 14))
    db_g, _ = ccdf(g)
    assert db_g.max() > 6.0
