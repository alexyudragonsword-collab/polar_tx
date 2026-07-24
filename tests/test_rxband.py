"""RX-band noise: analytic budgets behave physically."""
import numpy as np

from polartx.metrics.rxband import adpll_rxband, dtc_rxband
from polartx.presets import lte20_adpll, wifi_dtc


def test_adpll_rxband_matches_analyze():
    pm = lte20_adpll().tx.phasemod
    out = adpll_rxband(pm, offsets_hz=(20e6, 45e6))
    ana = pm.analyze()
    s = np.interp(45e6, ana.f, ana.pn_breakdown["total"])
    assert abs(out[45e6]["ldbc_hz"] - 10 * np.log10(s / 2)) < 1.0
    assert out[45e6]["ldbc_hz"] < out[20e6]["ldbc_hz"]   # falls with offset


def test_dtc_rxband_components():
    cfg = wifi_dtc(bw=160e6).tx.phasemod.cfg
    out = dtc_rxband(cfg, 640e6, offsets_hz=(45e6, 120e6))
    assert -170.0 < out[45e6]["ldbc_hz"] < -120.0
    # dither-shaped quantization rises with offset; LO Leeson falls -
    # both present, so just require sane bounded levels
    assert -170.0 < out[120e6]["ldbc_hz"] < -120.0
