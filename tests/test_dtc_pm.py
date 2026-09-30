"""Open-loop DTC phase modulator: quantization, dither, INL spurs, ZOH."""
import numpy as np
import pytest

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


def test_lo_pn_seed_pins_the_lo_sample_independently_of_the_run_seed():
    """One LO shared by two modulators: with ``lo_pn_seed`` set, two runs
    with different run seeds draw the IDENTICAL LO phase-noise sample (no
    jitter here, so the outputs are equal); with it unset the LO follows
    the run seed and the two differ.  Jitter, when on, must stay on the
    run seed: the difference between the two runs is then white."""
    from polartx.vendor.pllsim.blocks.oscillator import OscConfig
    lo = OscConfig(f0=5.9e9, gain=1.0, pn_dbchz=-115.0, pn_foffset=1e6,
                   pn_f1f3=2e5, pn_floor_dbchz=-155.0)
    ph = _cw_phase(10e6, n=1 << 14)
    pinned = DTCPhaseModulator(DTCPMConfig(n_bits=11, lo_pn=lo, lo_loop_bw=4e5,
                                           lo_pn_seed=7))
    a = pinned.modulate(ph, FS, noise=True, seed=0).phase_out
    b = pinned.modulate(ph, FS, noise=True, seed=1).phase_out
    assert np.array_equal(a, b)
    assert np.std(a - ph) > 1e-4                         # the LO noise is there
    free = DTCPhaseModulator(DTCPMConfig(n_bits=11, lo_pn=lo, lo_loop_bw=4e5))
    assert not np.array_equal(free.modulate(ph, FS, noise=True, seed=0).phase_out,
                              free.modulate(ph, FS, noise=True, seed=1).phase_out)
    # unset -> bit-identical to the pre-existing single-LO behaviour: the
    # LO draw follows the jitter draw on the run seed
    same = DTCPhaseModulator(DTCPMConfig(n_bits=11, lo_pn=lo, lo_loop_bw=4e5,
                                         lo_pn_seed=None))
    assert np.array_equal(free.modulate(ph, FS, noise=True, seed=3).phase_out,
                          same.modulate(ph, FS, noise=True, seed=3).phase_out)
    # jitter stays per run: two runs sharing the LO differ by white noise
    # of variance 2 sigma^2
    tau, fout = 50e-15, 5.9e9
    jit = DTCPhaseModulator(DTCPMConfig(n_bits=11, lo_pn=lo, lo_loop_bw=4e5,
                                        lo_pn_seed=7, jitter_rms_s=tau, fout=fout))
    d = (jit.modulate(ph, FS, noise=True, seed=0).phase_out
         - jit.modulate(ph, FS, noise=True, seed=1).phase_out)
    assert np.std(d) == pytest.approx(np.sqrt(2) * TWOPI * fout * tau, rel=0.05)
