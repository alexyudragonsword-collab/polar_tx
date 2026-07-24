"""Waveform-fidelity upgrades: SC-FDMA, BLE cert metrics, linearized
GMSK, receiver-style EVM."""
import numpy as np
import pytest

from polartx.metrics.ble_metrics import freq_deviation
from polartx.metrics.ofdm_rx import evm_rx
from polartx.presets import ble_1m_adpll, bench_edge_polar_staszewski05
from polartx.vendor.padpd.metrics import evm_of_signal
from polartx.vendor.padpd.waveform.ofdm import generate_ofdm
from polartx.waveforms.ble import gfsk_ble
from polartx.waveforms.edr import edge_waveform, linearized_gmsk_pulse
from polartx.waveforms.ofdm import (GenOFDMConfig, lte_waveform,
                                    ofdm_waveform, wifi_waveform)


# ------------------------------------------------------- F1: SC-FDMA
def test_scfdma_papr_and_loopback():
    w0 = lte_waveform(20e6, 64, n_symbols=8, seed=0)
    w1 = lte_waveform(20e6, 64, n_symbols=8, seed=0, sc_fdma=True)
    assert w0.meta["papr_db"] - w1.meta["papr_db"] > 1.0   # ~1.7 dB
    assert evm_of_signal(w1.x, w1.ofdm_ref).db < -80.0     # exact loopback


def test_plain_path_still_bit_exact():
    cfg = GenOFDMConfig(bandwidth_hz=20e6, qam_order=256, n_symbols=3,
                        seed=9)
    assert np.array_equal(ofdm_waveform(cfg).x, generate_ofdm(cfg).x)


def test_scfdma_rejects_scattered_pilots():
    with pytest.raises(ValueError):
        ofdm_waveform(GenOFDMConfig(bandwidth_hz=20e6, dft_precode=True,
                                    n_pilots=8))


# ------------------------------------------- F2: BLE cert-grade metrics
def test_df2max_per_symbol_and_drift():
    p = ble_1m_adpll()
    wf = p.make_waveform(n_bits=600, pattern="10101010")
    d = freq_deviation(p.tx.run(wf, noise=True, seed=1).y, wf)
    assert d["frac_above_185k"] >= 0.999          # RF-PHY criterion
    assert d["df2max_p001_hz"] > 185e3
    assert abs(d["drift_hz_per_us"]) < 20.0       # no drift source modeled


def test_mod_index_tolerance_sweep():
    """h inside the spec tolerance 0.45-0.55 keeps delta-f1-avg inside
    the 225-275 kHz certification window (interior points: the window
    edges map exactly to the h limits, so boundary h sits ON the
    limit)."""
    for h in (0.46, 0.5, 0.54):
        wf = gfsk_ble(400, 32e6, 1e6, pattern="11110000", mod_index=h)
        from polartx.chain import ChainConfig, PolarTX
        from polartx.dpa import DPA, DPAConfig
        from polartx.phasemod import IdealPhaseModulator
        tx = PolarTX(ChainConfig(), IdealPhaseModulator(),
                     DPA(DPAConfig(n_bits=10)))
        d = freq_deviation(tx.run(wf, noise=False).y, wf)
        assert 225e3 <= d["df1_avg_hz"] <= 275e3


# ------------------------------------------- F3: linearized GMSK (EDGE)
def test_c0_extraction_self_check():
    _, nmse = linearized_gmsk_pulse(16)
    assert nmse < -17.0             # C0 carries ~99% of the GMSK energy


def test_lgmsk_spectrum_is_gmsk_like():
    """The real EDGE skirt: C0 is time-limited, so its spectrum is far
    WIDER than the old stylized SRRC(0.3) at 400 kHz — the SRRC version
    was unrealistically clean."""
    from polartx.vendor.padpd.metrics import psd
    lev = {}
    for pulse in ("srrc", "lgmsk"):
        w = edge_waveform(300, pulse=pulse, seed=1)
        f, pdb = psd(w.x, w.fs, nfft=1 << 14)
        m = np.abs(np.abs(f) - 400e3) < 20e3
        lev[pulse] = pdb[m].mean()
    assert lev["lgmsk"] > lev["srrc"] + 20.0


def test_edge_benchmark_with_lgmsk_still_in_class():
    """Non-Nyquist pulse: the metric references the ideal waveform (the
    EDGE spec's convention), so the pulse's own ISI is not punished."""
    p = bench_edge_polar_staszewski05()
    r0 = p.tx.run(p.make_waveform(300, seed=1), noise=False)
    assert r0.evm()["devm_pct"] < 0.1            # metric floor
    assert r0.evm()["reference"] == "ideal_waveform"
    r = p.tx.run(p.make_waveform(300, seed=1), noise=True, seed=1)
    assert 1.0 < r.evm()["devm_pct"] < 3.5


# --------------------------------- F4: receiver-style EVM (pilots etc.)
def test_tracked_evm_recovers_cpe_and_channel():
    """Per-symbol CPE + a linear channel: the plain scalar EVM tanks,
    the preamble+pilot receiver recovers both (deterministic, so the
    recovery is essentially exact)."""
    w = wifi_waveform(80e6, 256, n_symbols=6, seed=1, pilots=True,
                      preamble=2)
    cfg = w.ofdm_ref.config
    sym_len = (cfg.fft_size + cfg.cp_len) * cfg.oversampling
    rng = np.random.default_rng(0)
    cpe = np.repeat(np.cumsum(rng.normal(0, 0.05, w.n // sym_len + 1)),
                    sym_len)[:w.n]
    fir = np.array([1.0, 0.1 + 0.05j])
    y = np.convolve(w.x * np.exp(1j * cpe), fir, mode="full")[:w.n]
    assert evm_of_signal(y, w.ofdm_ref).db > -30.0
    assert evm_rx(y, w).db < -60.0


def test_tracked_evm_loopback_exact():
    w = wifi_waveform(20e6, 64, n_symbols=4, seed=2, pilots=True,
                      preamble=1)
    assert evm_rx(w.x, w).db < -80.0
