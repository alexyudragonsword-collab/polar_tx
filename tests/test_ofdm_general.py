"""Generalized OFDM engine: WiFi presets bit-exact vs the vendored padpd
generator; non-WiFi numerologies demodulate cleanly."""
import numpy as np

from polartx.vendor.padpd.metrics import evm_of_signal
from polartx.vendor.padpd.waveform import OFDMConfig, generate_ofdm
from polartx.waveforms.ofdm import GenOFDMConfig, ofdm_waveform


def test_wifi_bit_exact_vs_vendored():
    for bw in (20e6, 160e6):
        ref = generate_ofdm(OFDMConfig(bandwidth_hz=bw, qam_order=1024,
                                       n_symbols=3, seed=9))
        gen = generate_ofdm(GenOFDMConfig(bandwidth_hz=bw, qam_order=1024,
                                          n_symbols=3, seed=9))
        assert np.array_equal(ref.x, gen.x)
        assert np.array_equal(ref.tx_symbols, gen.tx_symbols)


def test_nr_style_numerology_loopback():
    """5G-NR-style 30 kHz SCS at 60 MHz: generation/demod stay exact with
    a non-WiFi spacing (stylized numerology, 94% occupancy fallback)."""
    wf = ofdm_waveform(GenOFDMConfig(bandwidth_hz=60e6, scs_hz=30e3,
                                     qam_order=256, n_symbols=2, seed=2))
    assert wf.ofdm_ref.config.fft_size == 2000
    assert evm_of_signal(wf.x, wf.ofdm_ref).db < -80.0
