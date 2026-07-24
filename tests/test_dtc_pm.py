"""Open-loop DTC phase modulator: quantization, dither, INL spurs, ZOH."""
import numpy as np

from polartx.analysis.responses import (dtc_quant_phase_rms, inl_sin_spur_dbc,
                                        zoh_image_dbc)
from polartx.phasemod import DTCPhaseModulator, DTCPMConfig

TWOPI = 2.0 * np.pi
FS = 640e6


def _cw_phase(f_off, n=1 << 16, fs=FS):
    return TWOPI * f_off * np.arange(n) / fs


def _spectrum_dbc(y):
    """FFT magnitude spectrum normalized to the strongest bin [dBc]."""
    s = np.abs(np.fft.fft(y * np.hanning(y.size)))
    return 20 * np.log10(np.maximum(s / s.max(), 1e-12))


def test_quant_floor_law():
    """Round-quantizer phase error rms = LSB/sqrt(12) (B = 8..12)."""
    rng = np.random.default_rng(0)
    cmd = np.cumsum(rng.uniform(-1.0, 1.0, 1 << 15))     # code-rich trajectory
    for bits in (8, 10, 12):
        pm = DTCPhaseModulator(DTCPMConfig(n_bits=bits, dither=False))
        out = pm.modulate(cmd, FS, noise=False).phase_out
        meas = np.std(out - cmd)
        expect = dtc_quant_phase_rms(bits, osr=1.0)
        assert abs(20 * np.log10(meas / expect)) < 1.0


def test_dither_shapes_quant_noise_out_of_band():
    """First-order error feedback moves quantization power out of the
    low-frequency band: >= 5 dB in-band improvement at osr = 8."""
    rng = np.random.default_rng(1)
    n = 1 << 16
    cmd = np.cumsum(rng.uniform(-0.5, 0.5, n))
    err = {}
    for dither in (False, True):
        pm = DTCPhaseModulator(DTCPMConfig(n_bits=8, dither=dither))
        e = pm.modulate(cmd, FS, noise=False).phase_out - cmd
        spec = np.abs(np.fft.rfft(e)) ** 2
        err[dither] = spec[1:n // 16].sum()               # f < fs/8 band
    gain_db = 10 * np.log10(err[False] / err[True])
    assert gain_db > 5.0


def test_inl_sin_spur_level():
    """Sinusoidal INL (k cycles over the range) under a CW offset
    stimulus makes sidebands at k*f_off at 20log10(2*pi*amp/2) dBc."""
    amp_ui, k, f_off = 2e-3, 3, 5e6
    pm = DTCPhaseModulator(DTCPMConfig(n_bits=14, inl_sin=(amp_ui, k, 0.0)))
    out = pm.modulate(_cw_phase(f_off), FS, noise=False).phase_out
    spec = _spectrum_dbc(np.exp(1j * out))
    n = out.size
    spur_bin = int(round((f_off + k * f_off) * n / FS))
    meas = spec[spur_bin - 2: spur_bin + 3].max()
    assert abs(meas - inl_sin_spur_dbc(amp_ui)) < 2.0


def test_zoh_update_clock_images():
    """Phase update at fs/4 replicates the +f_sig tone at
    f_sig - f_update (the k = -1 sampling image) at the
    ZOH-sinc-predicted level."""
    f_sig, hold = 10e6, 4
    f_up = FS / hold
    pm = DTCPhaseModulator(DTCPMConfig(n_bits=14, f_update=f_up))
    out = pm.modulate(_cw_phase(f_sig), FS, noise=False).phase_out
    spec = _spectrum_dbc(np.exp(1j * out))
    n = out.size
    img_bin = n - int(round((f_up - f_sig) * n / FS))   # negative frequency
    meas = spec[img_bin - 2: img_bin + 3].max()
    assert abs(meas - zoh_image_dbc(f_sig, f_up)) < 2.0
