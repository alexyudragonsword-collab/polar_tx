"""Headless GUI compute layer: every preset builds and reports."""
import matplotlib

matplotlib.use("Agg")

import pytest

from polartx.guiutil import PRESETS, build_preset, run_chain_report


def test_all_presets_build():
    for name in PRESETS:
        p = build_preset(name)
        assert p.fs_bb > 0 and callable(p.make_waveform)


@pytest.mark.parametrize("name", ["BLE LE-1M", "BT EDR3 8DPSK",
                                  "WiFi 80 MHz"])
def test_report_has_metrics_and_figure(name):
    rep = run_chain_report(name, seed=1, noise=True, n_units=200)
    assert rep["metrics"].get("mask") in ("PASS", "FAIL")
    assert rep["fig"] is not None
    import matplotlib.pyplot as plt
    plt.close(rep["fig"])
