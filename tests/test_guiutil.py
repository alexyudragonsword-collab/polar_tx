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
    from polartx.presets import ble_1m_adpll

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


# ------------------------------------------------ which knobs reach which chain
def test_every_registered_preset_declares_its_knobs():
    """A preset without an entry would silently get no knobs at all; the
    benchmarks are the only ones meant to have none."""
    from polartx.guiutil import DEFAULT_PRESET, honoured_knobs
    assert DEFAULT_PRESET in PRESETS
    for name in PRESETS:
        if name.startswith("Bench:"):
            assert honoured_knobs(name) == ()
        else:
            assert honoured_knobs(name), name
    # that every one of them acts on its chain is tested on the full knob
    # specs (test_guispec.py), of which this legacy table is a subset


def test_chain_overrides_keeps_the_dtc_knob_off_the_rf_dac():
    """The RF-DAC preset's n_bits is its DAC resolution (12); the
    workbench's DTC resolution knob (11) used to land there."""
    from polartx.guiutil import chain_overrides
    ov = chain_overrides("WiFi 160 MHz (RF-DAC)", n_bits=11, env_skew_s=1e-9,
                         cfr_papr_db=None)
    assert ov == {}
    assert build_preset("WiFi 160 MHz (RF-DAC)", **ov).tx.rfdac.cfg.n_bits == 12
    assert chain_overrides("WiFi 160 MHz", n_bits=11)["n_bits"] == 11
    assert "env_skew_s" not in chain_overrides("WiFi 160 MHz (outphasing)",
                                               env_skew_s=1e-9)
    assert "env_skew_s" not in chain_overrides("BLE LE-1M", env_skew_s=1e-9)
    assert chain_overrides("WiFi 160 MHz", cfr_off=True) == {"cfr_papr_db": None}
    assert chain_overrides("BLE LE-1M", cfr_off=True) == {}


# ------------------------------------------------- Bluetooth metric set
def test_bluetooth_reports_in_band_emission_not_a_generic_aclr():
    """The generic ACLR takes the adjacent channel at one waveform
    bandwidth, where GFSK's own spectrum still is: it read -22.2 dBc on
    BLE LE-1M while the same run, measured at the in-band emission
    offsets, is -56 / -61 dBc.  Bluetooth chains report the latter (and
    GFSK its frequency deviation); EDGE is not Bluetooth and keeps ACLR."""
    import matplotlib.pyplot as plt

    from polartx.metrics.ble_metrics import bt_acp, freq_deviation
    rep = run_chain_report("BLE LE-1M", seed=1)
    m = rep["metrics"]
    assert "ACLR upper [dBc]" not in m
    acp = bt_acp(rep["result"].y, rep["result"].fs)
    assert m["in-band +-2 MHz [dBc]"] == round(max(acp["acp+2MHz_dbc"],
                                                   acp["acp-2MHz_dbc"]), 1)
    assert m["in-band +-2 MHz [dBc]"] < -40.0                 # measured -56.0
    assert m["in-band +-3 MHz [dBc]"] < m["in-band +-2 MHz [dBc]"]
    d = freq_deviation(rep["result"].y, rep["waveform"])
    assert m["freq deviation avg [kHz]"] == round(d["dev_avg_hz"] / 1e3, 1)
    assert "drift [Hz/us]" in m
    # the right-hand panel is the frequency trajectory, not the unit circle
    assert "frequency" in rep["fig"].axes[1].get_title()
    plt.close(rep["fig"])

    # LE 2M: offsets on its 2 MHz raster, the convention ble_mask uses
    m2 = run_chain_report("BLE LE-2M", seed=1)["metrics"]
    assert "in-band +-4 MHz [dBc]" in m2 and "in-band +-6 MHz [dBc]" in m2

    edr = run_chain_report("BT EDR3 8DPSK", seed=1)["metrics"]
    assert "in-band +-2 MHz [dBc]" in edr and "ACLR upper [dBc]" not in edr
    edge = run_chain_report("Bench: Staszewski'05 EDGE", seed=1)["metrics"]
    assert "ACLR upper [dBc]" in edge and "in-band +-2 MHz [dBc]" not in edge
    plt.close("all")


# ------------------------------------------------------ calibration demos
def test_calibration_demos_recover_what_was_injected():
    """The three calibration demos moved out of the two GUIs' page code
    (written twice, testable only through a GUI) into guiutil."""
    from polartx.guiutil import run_dtc_cal, run_skew_cal, run_twopoint_cal
    for method in ("waveform align", "ACP power search"):
        r = run_skew_cal(2.0, method=method)
        assert r["error_ps"] < 10.0, r                          # measured 0.1 / 0.1 ps
    with pytest.raises(ValueError, match="estimator"):
        run_skew_cal(2.0, method="guess")
    t = run_twopoint_cal(3.0)
    assert abs(t["estimated_pct"] - 3.0) < 0.1                  # measured 3.00
    assert t["evm_after_db"] < t["evm_before_db"] - 20.0        # -21.6 -> -53.4
    d = run_dtc_cal(2)
    assert d["err_rms_after_mrad"] < 0.1 * d["err_rms_before_mrad"]   # 18.0 -> 0.57


def test_no_page_does_its_own_calibration_physics():
    """The pages lay out guiutil results; they must not import the cal
    modules themselves again."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    for page in [root / "gui" / "pages" / "2_Calibration.py",
                 root / "src" / "polartx" / "guiqt" / "pages.py"]:
        text = page.read_text(encoding="utf-8")
        assert "polartx.cal" not in text and "from ..cal" not in text, page


# ------------------------------------------------- sweeps and comparisons
def test_a_sweep_is_the_single_runs_point_for_point():
    """A sweep is a convenience, never a second implementation: each point
    must be exactly the report a user gets by setting that value by hand."""
    import matplotlib.pyplot as plt

    from polartx.guispec import overrides_from_ui
    from polartx.guiutil import run_sweep
    held = {"dtc_bits": 10}
    sw = run_sweep("WiFi 160 MHz", "skew_ns", values=held,
                   points=[0.0, 0.2], seed=3)
    assert sw["x"] == [0.0, 0.2]
    for i, v in enumerate(sw["x"]):
        single = run_chain_report(
            "WiFi 160 MHz", seed=3,
            **overrides_from_ui("WiFi 160 MHz", {**held, "skew_ns": v}))
        for k, vals in sw["metrics"].items():
            assert vals[i] == single["metrics"].get(k), (k, v)
        plt.close(single["fig"])
    # the convention residual appears only once it matters: None, then a number
    assert sw["metrics"]["EVM per-tone eq [dB]"][0] is None
    assert sw["metrics"]["EVM per-tone eq [dB]"][1] is not None
    assert set(sw["metric_info"]) == set(sw["metrics"])


def test_sweep_points_follow_the_knob():
    from polartx.guiutil import run_sweep, sweep_points
    assert sweep_points("WiFi 160 MHz", "dtc_bits") == list(range(6, 15))
    assert sweep_points("LTE 20 MHz", "mode") == ["response", "event"]
    assert sweep_points("WiFi 160 MHz", "dpd") == [False, True]
    pts = sweep_points("WiFi 160 MHz", "skew_ns")
    assert len(pts) == 11 and pts[0] == -4.0 and pts[-1] == 4.0
    with pytest.raises(ValueError):
        run_sweep("WiFi 160 MHz (RF-DAC)", "skew_ns")


def test_the_comparison_is_the_individual_runs_with_shared_controls():
    """Same seed, same burst, shared controls; each topology takes only
    the ids it offers — the DTC resolution does not reach the RF-DAC."""
    import matplotlib.pyplot as plt

    from polartx.guiutil import COMPARE_SET, run_compare
    cmp = run_compare(values={"dtc_bits": 9, "skew_ns": 0.5}, seed=2)
    assert cmp["names"] == list(COMPARE_SET)
    for n in COMPARE_SET:
        from polartx.guispec import overrides_from_ui
        ov = overrides_from_ui(n, {"dtc_bits": 9, "skew_ns": 0.5})
        single = run_chain_report(n, seed=2, **ov)
        assert cmp["metrics"][n] == single["metrics"], n
        plt.close(single["fig"])
    assert "n_bits" not in overrides_from_ui("WiFi 160 MHz (RF-DAC)",
                                             {"dtc_bits": 9})
    # the skew reaches only the polar chain, and that is where it bites
    m = cmp["metrics"]
    assert m["WiFi 160 MHz"]["EVM [dB]"] > m["WiFi 160 MHz (RF-DAC)"]["EVM [dB]"] + 10
    panel = cmp["plots"][0]
    assert [s.get("color") for s in panel["series"][:3]] == [0, 1, 2]
