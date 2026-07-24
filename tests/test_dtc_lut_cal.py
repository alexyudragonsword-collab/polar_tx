"""Open-loop DTC gain/INL calibration: spur kill and residual floor."""
import numpy as np

from polartx.cal.dtc_cal import apply_dtc_correction, fit_dtc_correction
from polartx.phasemod import DTCPhaseModulator, DTCPMConfig

FS = 640e6
N = 1 << 16


def _cw():
    return 2 * np.pi * 5e6 * np.arange(N) / FS


def _spur_dbc(pm, k=3, f_off=5e6):
    out = pm.modulate(_cw(), FS, noise=False).phase_out
    s = np.abs(np.fft.fft(np.exp(1j * out) * np.hanning(N)))
    db = 20 * np.log10(np.maximum(s / s.max(), 1e-12))
    b = int(round(k * f_off * N / FS))
    return db[b - 2: b + 3].max()


def test_cal_kills_inl_spur_and_gain():
    pm = DTCPhaseModulator(DTCPMConfig(
        n_bits=12, gain_error=0.01, inl_sin=(2e-3, 3, 0.0),
        inl_poly=(0.0, 1e-3, -2e-3)))
    before = _spur_dbc(pm)
    for _ in range(2):                       # two fit/apply iterations
        apply_dtc_correction(pm, fit_dtc_correction(pm, _cw(), FS))
    after = _spur_dbc(pm)
    assert after < before - 25.0             # measured: -47 -> -91 dBc
    fit = fit_dtc_correction(pm, _cw(), FS)
    assert abs(fit["gain_hat"]) < 5e-4       # gain error corrected
    # residual within 2x of the pure-quantization floor
    qfloor = 2 * np.pi / 2 ** 12 / np.sqrt(12)
    assert fit["err_rms_before"] < 2.0 * qfloor


def test_cal_noop_on_clean_dtc():
    pm = DTCPhaseModulator(DTCPMConfig(n_bits=12))
    fit = fit_dtc_correction(pm, _cw(), FS)
    assert abs(fit["gain_hat"]) < 5e-4
    assert np.max(np.abs(fit["inl_lut_rad"])) < 3e-4
