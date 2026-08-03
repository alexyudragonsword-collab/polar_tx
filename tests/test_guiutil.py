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


def test_latest_benchmark_is_reachable():
    """The newest benchmark (Borokhovich RFIC'26) is selectable like any
    other preset AND has its own richer page; the 802.11n anchor is in the
    registry too."""
    assert "Bench: Borokhovich'26 WiFi7 MLO (FIR)" in PRESETS
    assert "Bench: 802.11n polar (~2010)" in PRESETS
    from polartx.guiutil import run_fir_report
    assert callable(run_fir_report)


def test_fir_benchmark_runs_through_the_chain_report():
    """The dual-tap preset scores the full metric set through the ordinary
    report path — the thing that used to be impossible."""
    rep = run_chain_report("Bench: Borokhovich'26 WiFi7 MLO (FIR)", seed=1)
    m = rep["metrics"]
    assert m["EVM [dB]"] < -25 and m["mask"] in ("PASS", "FAIL")
    assert "ACLR upper [dBc]" in m
    import matplotlib.pyplot as plt
    plt.close(rep["fig"])


def test_fir_report_measures_the_notch():
    from polartx.guiutil import run_fir_report
    rep = run_fir_report(bw=40e6, notch_offset_hz=500e6, n_symbols=2)
    m = rep["metrics"]
    assert m["OOC suppression [dB]"] > 3.0        # the notch does something
    assert m["tap delay tau [ps]"] == pytest.approx(1000.0, rel=1e-6)
    assert m["EVM FIR [dB]"] < -20
    import matplotlib.pyplot as plt
    plt.close(rep["fig"])


def test_selector_report_recommends_and_charts():
    from polartx.guiutil import run_selector_report
    wide = run_selector_report(bw_hz=320e6, evm_db_max=-38.0)
    assert "dtc_open_loop" in wide["metrics"]["recommendation"] or \
        "DTC" in wide["metrics"]["recommendation"]
    assert wide["metrics"]["adpll_two_point"] == "excluded"
    narrow = run_selector_report(bw_hz=10e6, evm_db_max=-30.0,
                                 two_point_gain_match=2e-3)
    assert "ADPLL" in narrow["metrics"]["recommendation"]
    import matplotlib.pyplot as plt
    plt.close(wide["fig"]); plt.close(narrow["fig"])


def test_combiner_report_beats_single_core_on_efficiency():
    from polartx.guiutil import run_combiner_report
    rep = run_combiner_report(n_way=2, backoff_db=6.0, peaking="C")
    m = rep["metrics"]
    assert m["avg eff Doherty [%]"] > m["avg eff single-core SCPA [%]"]
    # a balanced combiner has no handoff distortion
    assert m["AM-AM ripple [dB]"] < 1e-6 and m["AM-PM pp [deg]"] < 1e-6
    dirty = run_combiner_report(n_way=2, gain_imbalance_pct=10.0,
                                phase_imbalance_deg=8.0)
    assert dirty["metrics"]["AM-PM pp [deg]"] > 1.0
    import matplotlib.pyplot as plt
    plt.close(rep["fig"]); plt.close(dirty["fig"])


def test_rtl_export_writes_datapath_and_ams(tmp_path):
    from polartx.guiutil import run_rtl_export
    rep = run_rtl_export(str(tmp_path), n_bits=8, n_thermo=5, verify=False)
    for f in ("cfr_clip.v", "dtc_phase_acc.v", "dpa_thermo_decode.v",
              "dpa_rnm.vams", "polar_dpd_lut.v"):
        assert f in rep["files"]
    assert "OK" in rep["checks"]["DPA RNM self-check"]


def test_sc_fdma_constellation_is_de_precoded():
    """SC-FDMA regression: the frequency-domain symbols are DFT-precoded and
    form a Gaussian blob; the QAM lattice only exists after the receiver's
    inverse DFT.  The report used to plot the precoded grid, which showed a
    meaningless cloud for every LTE preset."""
    import numpy as np
    from polartx.presets import lte20_adpll
    p = lte20_adpll(qam=64)
    wf = p.make_waveform(n_symbols=8, seed=0)
    assert wf.meta["dft_precode"] is True
    # precoded grid: many levels (a cloud).  De-precoded: a 64-QAM lattice.
    precoded = wf.ofdm_ref.tx_symbols
    assert len(np.unique(np.round(precoded.real, 3))) > 100
    assert len(np.unique(np.round(wf.meta["qam_symbols"].real, 3))) == 8

    rep = run_chain_report("LTE 20 MHz", seed=1)
    assert "de-precoded" in rep["fig"].axes[1].get_title()
    # and the plotted cloud must actually be the lattice
    pts = rep["fig"].axes[1].get_lines()[0].get_xdata()
    assert len(np.unique(np.round(pts, 1))) <= 12    # 8 levels + rounding
    import matplotlib.pyplot as plt
    plt.close(rep["fig"])


def test_report_flags_why_a_constellation_is_fuzzy():
    """A fuzzy picture has two different causes; the report must say which."""
    clean = run_chain_report("WiFi 160 MHz", seed=1)
    assert clean["metrics"]["constellation"] == "resolved"
    dense = run_chain_report("Bench: Degani'24 WiFi7", seed=1)
    # 4096-QAM at its published -37 dB class: clouds genuinely overlap
    assert "overlap" in dense["metrics"]["constellation"]
    import matplotlib.pyplot as plt
    plt.close(clean["fig"]); plt.close(dense["fig"])


def test_cpe_is_reported_and_negligible_for_the_wideband_presets():
    """Neither scalar nor per_tone equalization removes common phase error
    (per_tone averages ALONG the symbol axis), so the report states the
    residual rather than leaving the convention implicit.

    It is ~0.1 deg here because these presets run a PLL-locked LO whose
    noise is high-pass shaped above the OFDM symbol rate: little energy
    lands in the common-phase term, and what remains is fast (ICI), which
    pilot CPE tracking could not remove either."""
    import matplotlib.pyplot as plt
    for name in ("WiFi 160 MHz", "Bench: Degani'24 WiFi7"):
        rep = run_chain_report(name, seed=1)
        assert rep["metrics"]["CPE rms [deg]"] < 0.5
        # negligible -> no "would be better with CPE tracking" line
        assert "EVM if CPE tracked [dB]" not in rep["metrics"]
        plt.close(rep["fig"])


def test_cpe_detector_catches_an_injected_common_phase():
    """Guards the above: a CPE report that always says ~0 would be
    indistinguishable from a broken detector.  Stamp a known per-symbol
    common phase on the chain output and require it to be seen."""
    import numpy as np
    import matplotlib.pyplot as plt
    from polartx.presets import wifi_dtc
    from polartx.waveforms.ofdm import demodulate_ofdm
    p = wifi_dtc(bw=80e6)
    wf = p.make_waveform()
    res = p.tx.run(wf, noise=True, seed=1)
    y = res.y.copy()
    nsym = wf.ofdm_ref.tx_symbols.shape[0]
    seg = len(y) // nsym
    rng = np.random.default_rng(3)
    ph = np.deg2rad(rng.normal(0.0, 3.0, nsym))
    for i in range(nsym):
        y[i * seg:(i + 1) * seg] *= np.exp(1j * ph[i])
    rx = demodulate_ofdm(y, wf.ofdm_ref)
    tx = wf.ofdm_ref.tx_symbols
    g = (np.conj(tx) * rx).sum(axis=0) / (np.abs(tx) ** 2).sum(axis=0)
    eq = rx / g
    cpe = np.angle((eq * np.conj(tx)).sum(axis=1, keepdims=True))
    assert np.rad2deg(cpe.std()) > 2.0            # the injection is seen
    e0 = np.sqrt((np.abs(eq - tx) ** 2).mean())
    e1 = np.sqrt((np.abs(eq * np.exp(-1j * cpe) - tx) ** 2).mean())
    assert 20 * np.log10(e0 / e1) > 5.0           # removing it recovers EVM
    plt.close("all")


def test_skew_damage_is_not_common_phase():
    """AM/PM path skew wrecks EVM fast, but NOT via CPE.

    Skewing the envelope against the phase gives y = x * a(t-tau)/a(t), a
    real, data-dependent, broadband gain — spread distortion (ICI), not a
    per-symbol common rotation.  So CPE tracking buys nothing no matter
    how bad the skew gets, while the EVM collapses."""
    import numpy as np
    import matplotlib.pyplot as plt
    prev = 0.0
    for sk_ns in (0.0, 0.5, 2.0):
        rep = run_chain_report("WiFi 160 MHz", seed=1,
                               env_skew_s=sk_ns * 1e-9)
        m = rep["metrics"]
        assert m["CPE rms [deg]"] < 0.5           # CPE stays negligible
        assert "EVM if CPE tracked [dB]" not in m  # tracking would buy <0.5 dB
        if sk_ns:
            assert m["EVM [dB]"] > prev + 5.0      # yet EVM collapses fast
        prev = m["EVM [dB]"]
        plt.close(rep["fig"])


def test_skew_damage_is_partly_equalizable():
    """The part of the skew damage a receiver DOES remove is the linear
    frequency response, not CPE — several dB at 0.5 ns, which is why the
    report surfaces the per-tone number once it matters."""
    import matplotlib.pyplot as plt
    clean = run_chain_report("WiFi 160 MHz", seed=1)
    assert "EVM per-tone eq [dB]" not in clean["metrics"]
    skewed = run_chain_report("WiFi 160 MHz", seed=1, env_skew_s=0.5e-9)
    m = skewed["metrics"]
    assert "EVM per-tone eq [dB]" in m
    # equalizable part is several dB, and does NOT rescue the link
    assert m["EVM per-tone eq [dB]"] - m["EVM [dB]"] < -3.0
    assert m["EVM per-tone eq [dB]"] > -35.0
    plt.close(clean["fig"]); plt.close(skewed["fig"])


def test_power_domain_catches_skew_before_evm_does():
    """Instrument correspondence, pinned.

    A spectrum analyser has no equalizer, so AM/PM skew's spectral
    regrowth is fully exposed there — and it alarms EARLIER than the
    VSA-convention (per-tone) EVM does.  This is the ex05 lesson in the
    OFDM setting: EVM can look survivable while ACP/mask already fail."""
    import numpy as np
    import matplotlib.pyplot as plt
    from polartx.metrics import check_mask
    from polartx.metrics.masks import default_mask
    from polartx.presets import wifi_dtc
    from polartx.waveforms.ofdm import demodulate_ofdm

    def run(sk_ns):
        p = wifi_dtc(bw=160e6, env_skew_s=sk_ns * 1e-9)
        wf = p.make_waveform(n_symbols=8, seed=0)
        res = p.tx.run(wf, noise=True, seed=1)
        rx = demodulate_ofdm(res.y, wf.ofdm_ref)
        tx = wf.ofdm_ref.tx_symbols
        g = (np.conj(tx) * rx).sum(axis=0) / (np.abs(tx) ** 2).sum(axis=0)
        evm = 20 * np.log10(np.sqrt((np.abs(rx / g - tx) ** 2).mean()
                                    / (np.abs(tx) ** 2).mean()))
        f, pdb = res.psd(nfft=8192)
        ok, _, _ = check_mask(f, pdb, default_mask(wf))
        return evm, res.aclr()["upper_dbc"], ok

    e0, a0, ok0 = run(0.0)
    assert ok0 and a0 < -50                    # clean chain passes both
    e2, a2, ok2 = run(0.2)
    # at 0.2 ns the instrument-convention EVM is still usable for 1024-QAM
    assert e2 < -33.0
    # ...but the power-domain measurement has already failed hard
    assert not ok2 and a2 > a0 + 15.0
    plt.close("all")


def test_every_preset_accepts_am_pm_skew():
    """AM/PM skew is an envelope-path impairment and the chain applies it
    architecture-agnostically, but only the wideband presets used to
    expose it — the narrowband ones raised TypeError, so the GUI could
    never sweep skew on LTE/EDR/BLE."""
    import matplotlib.pyplot as plt
    for name in ("BLE LE-1M", "BT EDR3 8DPSK", "LTE 20 MHz",
                 "WiFi 160 MHz", "NR FR1 100 MHz"):
        rep = run_chain_report(name, seed=1, env_skew_s=1e-9)
        assert rep["metrics"]["mask"] in ("PASS", "FAIL")
        plt.close(rep["fig"])


def test_skew_sensitivity_follows_the_envelope_not_the_architecture():
    """The narrowband ADPLL chain is NOT skew-immune: what makes a polar
    TX skew-sensitive is envelope variation, so LTE's OFDM payload is hit
    hard (just ~8x more tolerantly than 160 MHz WiFi, in proportion to
    bandwidth), while constant-envelope BLE is immune outright."""
    import numpy as np
    import matplotlib.pyplot as plt
    from polartx.chain import ChainConfig
    from polartx.presets import ble_1m_adpll, lte20_adpll

    # BLE GFSK: env == 1, so skew is mathematically a no-op.  Compare the
    # chain output itself, not a rounded metric.
    y0 = ble_1m_adpll(env_skew_s=0.0).tx.run(
        ble_1m_adpll().make_waveform(n_bits=200), noise=False, seed=1).y
    y1 = ble_1m_adpll(env_skew_s=20e-9).tx.run(
        ble_1m_adpll().make_waveform(n_bits=200), noise=False, seed=1).y
    assert np.allclose(y0, y1, atol=1e-12)

    # LTE OFDM: genuinely degrades, and the mask goes first (power domain)
    clean = run_chain_report("LTE 20 MHz", seed=1)
    skewed = run_chain_report("LTE 20 MHz", seed=1, env_skew_s=1e-9)
    assert clean["metrics"]["mask"] == "PASS"
    assert skewed["metrics"]["mask"] == "FAIL"
    assert skewed["metrics"]["EVM [dB]"] > clean["metrics"]["EVM [dB]"] + 10
    plt.close("all")
