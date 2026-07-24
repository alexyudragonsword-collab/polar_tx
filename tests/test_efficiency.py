"""DPA drain-efficiency model: laws and modulated averages."""
import numpy as np
import pytest

from polartx.chain import ChainConfig, PolarTX
from polartx.dpa import DPA, DPAConfig
from polartx.dpa.characteristics import efficiency_curve
from polartx.phasemod import IdealPhaseModulator
from polartx.waveforms.ofdm import wifi_waveform


def test_scpa_law_shape():
    x = np.linspace(0.0, 1.0, 200)
    eta = efficiency_curve(("scpa", 0.67, 0.85), x)
    assert eta[-1] == pytest.approx(0.85)          # peak at full scale
    assert eta[0] == 0.0
    assert (np.diff(eta) >= -1e-12).all()          # monotone
    # gamma=0.67 -> 60% of peak at half amplitude
    assert efficiency_curve(("scpa", 0.67, 1.0), np.array([0.5]))[0] == \
        pytest.approx(1.0 / 1.67, rel=1e-6)


def test_scpa_beats_classb_at_backoff():
    """The polar headline: at OFDM backoff the SCPA's quadratic loss law
    beats a class-B linear PA with the same peak efficiency."""
    x = np.linspace(0.05, 0.6, 100)                # deep-backoff region
    scpa = efficiency_curve(("scpa", 0.67, 0.85), x)
    clb = efficiency_curve(("classb", 0.85), x)
    assert (scpa > clb).mean() > 0.9


def test_constant_envelope_average_is_spot_value():
    dpa = DPA(DPAConfig(n_bits=10))
    code = np.full(1000, 700)
    spot = float(dpa.efficiency(code[:1])[0])
    assert dpa.average_efficiency(code)["eta_avg"] == pytest.approx(spot)


def test_cfr_improves_modulated_efficiency():
    """Clipping the PAPR lets the average envelope sit closer to full
    scale for the same peak: eta_avg rises."""
    wf = wifi_waveform(80e6, 256, n_symbols=4, seed=1)
    out = {}
    for papr in (None, 8.5):
        tx = PolarTX(ChainConfig(env_floor=0.02, cfr_papr_db=papr),
                     IdealPhaseModulator(), DPA(DPAConfig(n_bits=10)))
        res = tx.run(wf, noise=False)
        out[papr] = res.avg_efficiency(tx.dpa)["eta_avg"]
    assert out[8.5] > out[None] * 1.05
