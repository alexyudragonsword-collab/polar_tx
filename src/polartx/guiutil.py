"""Pure-computation layer behind the Streamlit GUI (testable headless).

Every function returns plain data + matplotlib figures; the GUI pages
only lay them out.  Mirrors the sibling repos' gui_core/services split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _registry():
    """Display name -> factory(**overrides) -> TxPreset.

    Standard chains plus the literature-class benchmarks (pllsim ex14
    convention); benchmarks ignore overrides that would break their
    published-class assumptions by fixing their own parameters."""
    from . import presets as P
    return {
        "BLE LE-1M": lambda **o: P.ble_1m_adpll(**o),
        "BLE LE-2M": lambda **o: P.ble_2m_adpll(**o),
        "BT EDR2 pi/4-DQPSK": lambda **o: P.bt_edr_adpll("pi4dqpsk", **o),
        "BT EDR3 8DPSK": lambda **o: P.bt_edr_adpll("8dpsk", **o),
        "LTE 20 MHz": lambda **o: P.lte20_adpll(**o),
        "WiFi 80 MHz": lambda **o: P.wifi_dtc(bw=80e6, **o),
        "WiFi 160 MHz": lambda **o: P.wifi_dtc(bw=160e6, **o),
        "WiFi 320 MHz": lambda **o: P.wifi_dtc(bw=320e6, **o),
        "NR FR1 100 MHz": lambda **o: P.nr_dtc(bw=100e6, **o),
        "NR FR2 200 MHz": lambda **o: P.nr_dtc(bw=200e6, **o),
        # outphasing topology on the WiFi 160 MHz plan: an OutphasingTxPreset,
        # interchangeable here via its .tx alias (same mechanism as the FIR
        # bench preset).  Not "Bench:" — it is a standard chain, A/B'd
        # against "WiFi 160 MHz" on the same waveform.
        "WiFi 160 MHz (outphasing)":
            lambda **o: P.wifi_outphasing(bw=160e6, **o),
        # digital I/Q (RF-DAC) topology on the same plan; a CartesianTxPreset
        # via its .tx alias.  The shared "n_bits" knob means DAC bits here.
        "WiFi 160 MHz (RF-DAC)":
            lambda **o: P.wifi_rfdac(bw=160e6, **o),
        "Bench: Staszewski'05 EDGE": lambda **o: P.bench_edge_polar_staszewski05(),
        "Bench: Madoglio'14 LTE-20": lambda **o: P.bench_lte20_polar_madoglio14(),
        "Bench: BenBassat'20 WiFi6": lambda **o: P.bench_wifi6_polar_benbassat20(),
        "Bench: Degani'24 WiFi7": lambda **o: P.bench_wifi7_polar_degani24(),
        # dual-tap FIR chain: a FIRTxPreset, interchangeable here via its
        # .tx alias.  The OOC-noise notch — the paper's headline — needs
        # the single-tap baseline to measure against, so the dedicated
        # page (run_fir_report) is still the fuller view.
        "Bench: Borokhovich'26 WiFi7 MLO (FIR)":
            lambda **o: P.bench_wifi7_mlo_fir_borokhovich26(),
        "Bench: 802.11n polar (~2010)": lambda **o: P.bench_wifi11n_polar(),
    }


#: chain presets selectable in the GUIs and the headless report layer
PRESETS = list(_registry())

#: the preset every front end opens on — by name, so reordering the
#: registry cannot silently move it (it was ``index=6`` in two places)
DEFAULT_PRESET = "WiFi 160 MHz"

# Which of the workbench's shared knobs each preset honours, keyed by the
# override keyword it is forwarded as.  One table instead of three front
# ends each guessing from the preset's name: that guessing sent the DTC
# resolution knob to the RF-DAC chain as its DAC resolution (11 instead of
# its own 12 bits) and offered an AM/PM skew on chains with no envelope
# path (outphasing, RF-DAC) or no envelope variation (BLE GFSK, measured
# bit-identical at 10 ns).
_DTC_POLAR = ("n_bits", "env_skew_s", "cfr_papr_db")
_HONOURS: dict[str, tuple[str, ...]] = {
    "BLE LE-1M": ("dp_gain", "loop_bw"),
    "BLE LE-2M": ("dp_gain", "loop_bw"),
    "BT EDR2 pi/4-DQPSK": ("dp_gain", "loop_bw", "env_skew_s"),
    "BT EDR3 8DPSK": ("dp_gain", "loop_bw", "env_skew_s"),
    "LTE 20 MHz": ("dpd", "dp_gain", "env_skew_s"),
    "WiFi 80 MHz": _DTC_POLAR,
    "WiFi 160 MHz": _DTC_POLAR,
    "WiFi 320 MHz": _DTC_POLAR,
    "NR FR1 100 MHz": _DTC_POLAR,
    "NR FR2 200 MHz": _DTC_POLAR,
    "WiFi 160 MHz (outphasing)": ("n_bits", "cfr_papr_db"),
    # its own n_bits means DAC bits (12 by default); the shared DTC knob
    # must not reach it
    "WiFi 160 MHz (RF-DAC)": ("cfr_papr_db",),
}


def honoured_knobs(name: str) -> tuple[str, ...]:
    """The shared workbench knobs this preset honours (override keywords).
    Benchmarks honour none: they fix their published-class parameters."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}")
    return _HONOURS.get(name, ())


def chain_overrides(name: str, *, cfr_off: bool = False, **values) -> dict:
    """Reduce the workbench's knob values to the ones this preset honours.

    Every front end passes everything it shows and gets back exactly what
    the preset should receive, so a knob that does not apply cannot reach
    a chain that would misread or ignore it.  A value of ``None`` means
    "leave the preset's default".  ``cfr_off`` switches CFR off where the
    preset has it (``cfr_papr_db=None`` is a real setting, so it cannot
    double as "not set")."""
    keep = honoured_knobs(name)
    out = {k: v for k, v in values.items() if k in keep and v is not None}
    if cfr_off and "cfr_papr_db" in keep:
        out["cfr_papr_db"] = None
    return out


def build_preset(name: str, **overrides):
    """Preset registry -> TxPreset; overrides forwarded to the factory
    (benchmark presets fix their own parameters and ignore overrides)."""
    reg = _registry()
    if name not in reg:
        raise ValueError(f"unknown preset {name!r}")
    return reg[name](**overrides)


def _size_kwargs(make_waveform, n_units: int | None) -> dict:
    """Map a burst-length request onto whatever length kwarg this
    preset's make_waveform actually accepts (n_bits / n_syms /
    n_symbols) — robust to the per-preset waveform signature."""
    if n_units is None:
        return {}
    import inspect
    params = inspect.signature(make_waveform).parameters
    for kw in ("n_bits", "n_syms", "n_symbols"):
        if kw in params:
            return {kw: n_units}
    return {}


@dataclass
class _Constellation:
    """The constellation panel's data, equalized with the SAME convention as
    the printed EVM — the invariant `evm_equalize_default` exists to keep
    (CONTRIBUTING.md 5).  Computed with the metrics, drawn by the figure, so
    the picture cannot drift from the number."""

    pts: np.ndarray             # equalized symbols to scatter
    label: str                  # convention suffixes, as they appear in the title
    zoom_span: float | None     # dense-QAM centre zoom, None = full square


def _evm_metrics(e) -> dict:
    """The EVM entry for whichever metric family this waveform kind uses."""
    if hasattr(e, "db"):
        return {"EVM [dB]": round(e.db, 1), "EVM [%]": round(e.percent, 2)}
    if "devm_pct" in e:
        return {"DEVM [%]": round(e["devm_pct"], 2)}
    return {"phase EVM [%]": round(e["evm_pct"], 2)}


def _is_bluetooth(wf) -> bool:
    """BLE GFSK, or BR/EDR DPSK at 1 Msym/s (EDGE is DPSK too, at
    270.833 ksym/s, and is not Bluetooth)."""
    return wf.kind == "gfsk" or (wf.kind == "dpsk"
                                 and wf.meta.get("rate_sym") == 1e6)


def _bluetooth_metrics(res, wf) -> tuple[dict, np.ndarray | None]:
    """In-band emission (and, for GFSK, frequency deviation) in place of
    the generic ACLR.

    The generic ACLR takes the adjacent channel one waveform bandwidth
    away, where GFSK's own spectrum still is (-22 dBc on BLE LE-1M against
    -56 dBc at the in-band emission offset of the same run).  Offsets are
    the BR/EDR / LE-1M ±2 and ±3 MHz, on the 2 MHz raster for LE 2M —
    the scaling ``metrics.masks.ble_mask`` uses; like that mask and
    ``bt_acp`` itself this is a stylized template, not a conformance
    measurement.  The deviation figures are over this burst's own payload,
    not the certification patterns (11110000 / 10101010) that define
    delta-f1-avg and delta-f2-max.

    Returns the metrics and, for GFSK, the instantaneous-frequency trace
    the figure draws, so the panel shows the samples the numbers came
    from.
    """
    from .metrics.ble_metrics import TWOPI, bt_acp, freq_deviation
    s = wf.meta.get("rate", 1e6) / 1e6 if wf.kind == "gfsk" else 1.0
    offs = (2e6 * s, 3e6 * s)
    acp = bt_acp(res.y, res.fs, offsets_hz=offs, ch_bw=1e6 * s)
    metrics: dict = {}
    for off in offs:
        tag = f"{off / 1e6:g}MHz_dbc"
        worst = max(acp["acp+" + tag], acp["acp-" + tag])
        metrics[f"in-band +-{off / 1e6:g} MHz [dBc]"] = round(worst, 1)
    trace = None
    if wf.kind == "gfsk":
        d = freq_deviation(res.y, wf)
        metrics["freq deviation avg [kHz]"] = round(d["dev_avg_hz"] / 1e3, 1)
        metrics["drift [Hz/us]"] = round(d["drift_hz_per_us"], 1)
        trace = np.gradient(np.unwrap(np.angle(res.y))) * res.fs / TWOPI
    return metrics, trace


def _spectrum_metrics(res, wf) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """PSD, the mask trace to draw, and the two mask verdicts.

    Returns the spectrum arrays as well so the figure draws exactly the PSD
    the verdict was computed from, rather than a second, independent one.
    """
    from .metrics import check_mask
    from .metrics.masks import default_mask, default_mask_spec
    from .metrics.sem import check_sem

    f, pdb = res.psd(nfft=8192)
    ok, _margin, mask_db = check_mask(f, pdb, default_mask(wf))
    metrics = {"mask": "PASS" if ok else "FAIL"}
    # The plain margin is pinned at 0.00 dB whenever the signal passes (the
    # mask is 0 dBr in-channel and the PSD is peak-normalized, so the worst
    # point is the carrier tangency). Report the OUT-OF-BAND margin, which
    # is the number that actually moves with the design, measured the way a
    # spectrum analyser does — integrated in the mask's resolution
    # bandwidth rather than per FFT bin.
    _sem = check_sem(f, pdb, default_mask_spec(wf), channel_bw_hz=wf.bw)
    if "oob_margin_db" in _sem:
        metrics["mask OOB margin [dB]"] = round(_sem["oob_margin_db"], 1)
    return f, pdb, mask_db, metrics


def _convention_metrics(rx: np.ndarray, tx_s: np.ndarray, eq_grid: np.ndarray,
                        eq: str, evm_db: float) -> dict:
    """What this report's equalization convention costs, stated rather than
    left implicit — the two residuals a reader would otherwise have to know
    to ask about (CONTRIBUTING.md 5)."""
    metrics: dict = {}
    # Neither scalar nor per_tone equalization removes COMMON PHASE
    # ERROR: per_tone averages along the symbol axis, so a per-symbol
    # phase rotation survives it.  A real receiver tracks CPE off the
    # pilots, so the convention matters when comparing to measured
    # numbers — report the residual instead of leaving it implicit.
    # (These presets run a PLL-locked LO whose noise is high-pass
    # shaped above the symbol rate, so CPE is normally ~0.1 deg and
    # what phase noise remains is fast, i.e. ICI that CPE tracking
    # cannot fix either.)
    if eq_grid.ndim == 2 and eq_grid.shape[0] > 1:
        _cpe = np.angle((eq_grid * np.conj(tx_s)).sum(axis=1, keepdims=True))
        _res = eq_grid * np.exp(-1j * _cpe)
        _e0 = np.sqrt((np.abs(eq_grid - tx_s) ** 2).mean())
        _e1 = np.sqrt((np.abs(_res - tx_s) ** 2).mean())
        metrics["CPE rms [deg]"] = round(float(np.rad2deg(_cpe.std())), 2)
        _gain = 20 * np.log10(max(_e0, 1e-30) / max(_e1, 1e-30))
        if _gain > 0.5:
            metrics["EVM if CPE tracked [dB]"] = round(evm_db - _gain, 1)

    # The bigger convention effect is the LINEAR one.  Impairments like
    # AM/PM path skew produce a frequency-dependent response as well as
    # data-dependent distortion; scalar EVM (the default here, so that
    # memory distortion stays visible) counts the frequency response as
    # error, while a real receiver's per-tone equalizer removes it.  At
    # 0.5 ns of skew that split is ~5 dB — far larger than CPE — so
    # report it whenever it matters.
    if eq == "scalar" and rx.ndim == 2 and rx.shape[0] > 1:
        _gt = ((np.conj(tx_s) * rx).sum(axis=0)
               / (np.abs(tx_s) ** 2).sum(axis=0))
        _et = np.sqrt((np.abs(rx / _gt - tx_s) ** 2).mean()
                      / (np.abs(tx_s) ** 2).mean())
        _etdb = float(20 * np.log10(max(_et, 1e-30)))
        if evm_db - _etdb > 1.0:
            metrics["EVM per-tone eq [dB]"] = round(_etdb, 1)
    return metrics


def _ofdm_constellation(res, wf, e, eq: str,
                        evm_db: float) -> tuple[_Constellation, dict]:
    """Demodulate and equalize once, then hand back the panel and the numbers.

    Everything OFDM-specific about the report lives here: the equalizer, the
    convention residuals it implies, SC-FDMA de-precoding, and whether the
    lattice is resolvable at this EVM.  Nothing is demodulated twice, so the
    picture and the printed EVM cannot come from different equalizations.
    """
    from .waveforms.ofdm import demodulate_ofdm

    ref = wf.require_ofdm_ref()
    rx = demodulate_ofdm(res.y, ref)
    tx_s = ref.tx_symbols
    if eq == "per_tone" and rx.ndim == 2:
        # per-subcarrier channel estimate, exactly like the EVM metric:
        # removes the linear phase ramp (group delay) a receiver equalizes
        g = ((np.conj(tx_s) * rx).sum(axis=0)
             / (np.abs(tx_s) ** 2).sum(axis=0))
    else:
        g = np.vdot(tx_s, rx) / np.vdot(tx_s, tx_s)
    eq_grid = rx / g
    label = f"{eq} eq."
    metrics = _convention_metrics(rx, tx_s, eq_grid, eq, evm_db)

    qam_ref = wf.meta.get("qam_symbols")
    if wf.meta.get("dft_precode") and qam_ref is not None \
            and eq_grid.ndim == 2:
        # SC-FDMA: the frequency-domain symbols are DFT-precoded and
        # look like a Gaussian cloud — the QAM constellation only
        # exists after the receiver's inverse DFT across the
        # allocation.  Plotting the precoded grid shows a blob that
        # says nothing about link quality.
        n_act = eq_grid.shape[1]
        eq_grid = np.fft.ifft(eq_grid, axis=1) * np.sqrt(n_act)
        eq_grid = eq_grid[-qam_ref.shape[0]:]      # drop preamble rows
        label += ", DFT de-precoded"

    ref_pts = np.asarray(qam_ref if qam_ref is not None else tx_s)
    n_lvl = len(np.unique(np.round(ref_pts.real, 3)))
    # Why a constellation looks fuzzy has two distinct causes, and the
    # user cannot tell them apart by eye: either the error clouds are
    # genuinely wider than the lattice spacing (a link at its EVM
    # limit), or too few symbols were run to populate M sites. Say
    # which, so a legitimately marginal picture is not read as a bug.
    m_order = n_lvl ** 2
    if m_order > 1 and hasattr(e, "percent"):
        sep = (np.sqrt(6.0 / (m_order - 1)) / 2) / max(e.percent / 100, 1e-12)
        hits = ref_pts.size / m_order
        metrics["constellation"] = (
            "resolved" if sep > 3 else
            "marginal" if sep > 2 else "clouds overlap at this EVM")
        if hits < 4:
            metrics["constellation"] += f" (only {hits:.1f} pts/symbol site)"

    zoom_span = None
    if n_lvl >= 32:
        # 1024-QAM and denser: the full square is a solid wall of points
        # at any usable marker size.  Zoom to the centre sub-lattice so
        # the individual clouds — and whether they still separate at this
        # EVM — are actually visible.
        zoom_span = 4.5 * np.ptp(np.unique(np.round(ref_pts.real, 3))) / (n_lvl - 1)
        label += f", centre zoom ({n_lvl}^2-QAM)"
    return _Constellation(pts=eq_grid.ravel(), label=label,
                          zoom_span=zoom_span), metrics


def _chain_figure(name: str, wf, res, e, f: np.ndarray, pdb: np.ndarray,
                  mask_db: np.ndarray, cons: _Constellation | None,
                  f_trace: np.ndarray | None = None):
    """Spectrum + per-kind right-hand panel.  Rendering only: every number
    drawn here was computed by the helpers above."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(f / 1e6, pdb, lw=0.7)
    ax[0].plot(f / 1e6, mask_db, "r--", lw=1.0)
    ax[0].set(xlabel="offset [MHz]", ylabel="dBr", ylim=(-90, 5),
              title=f"{name} spectrum")
    if cons is not None:
        ax[1].plot(cons.pts.real, cons.pts.imag, ".", ms=1, alpha=0.4)
        if cons.zoom_span is not None:
            ax[1].set_xlim(-cons.zoom_span, cons.zoom_span)
            ax[1].set_ylim(-cons.zoom_span, cons.zoom_span)
        ax[1].set_title(f"constellation ({cons.label})")
    elif wf.kind == "dpsk":
        # trim the pulse-shaping edge transient, but scale the trim to the
        # burst: a fixed [20:-20] silently produced an EMPTY plot for any
        # burst under ~41 symbols
        _t = min(20, max(0, (e["symbols_rx"].size - 8) // 2))
        z = e["symbols_rx"][_t:-_t] if _t else e["symbols_rx"]
        ax[1].plot(z.real, z.imag, ".", ms=2, alpha=0.5)
        if wf.meta.get("pulse_taps") is not None:
            # EDGE 3pi/8-8PSK on the linearized-GMSK C0 pulse: C0 is
            # deliberately NOT Nyquist, so the symbol-instant samples carry
            # ISI by construction and never form tight clusters.  That is
            # why the DEVM for this pulse is scored against the ideal
            # waveform, not against these samples — the spread below is
            # the pulse, not a broken link.
            ax[1].set_title("8PSK symbols — linearized-GMSK C0\n"
                            "(non-Nyquist: ISI by design; DEVM uses the "
                            "ideal-waveform ref)", fontsize=8)
        else:
            ax[1].set_title(f"{wf.meta.get('mode', 'DPSK')} symbols")
    elif f_trace is not None:
        # GFSK is constant-envelope: its IQ trajectory is the unit circle
        # and says nothing.  What carries the modulation is the frequency.
        # 40 symbols from mid-burst: a PRBS starts from a sparse LFSR
        # state (the first ~100 bits of the default payload are nearly
        # all zeros), which would show one tone and no modulation
        sps = wf.fs / wf.meta["rate"]
        mid = f_trace.size // 2
        n0, n1 = max(0, mid - int(20 * sps)), min(f_trace.size,
                                                   mid + int(20 * sps))
        ax[1].plot((np.arange(n0, n1) - n0) / sps, f_trace[n0:n1] / 1e3,
                   lw=0.8)
        ax[1].set(xlabel="symbol", ylabel="kHz",
                  title="frequency deviation trajectory")
        ax[1].grid(True, alpha=0.3)
    else:
        y = res.y / np.abs(res.y).max()
        ax[1].plot(y.real[2000:12000], y.imag[2000:12000], lw=0.3, alpha=0.6)
        ax[1].set_title("IQ trajectory")
    if f_trace is None:
        ax[1].set_aspect("equal")
    fig.tight_layout()
    return fig


def run_chain_report(name: str, *, seed: int = 1, noise: bool = True,
                     n_units: int | None = None, **overrides) -> dict:
    """Build, run, and package metrics + figures for one chain.

    The three phases are separate on purpose: build/run, then every number,
    then the picture.  The numbers do not depend on the figure existing (the
    Android bridge and the tests want them without matplotlib), and the
    figure computes nothing of its own, so a panel cannot disagree with the
    metric printed beside it.
    """
    p = build_preset(name, **overrides)
    wf = p.make_waveform(**_size_kwargs(p.make_waveform, n_units))
    res = p.tx.run(wf, noise=noise, seed=seed)

    # one equalization convention for BOTH the number and the picture
    eq = getattr(res, "evm_equalize_default", "scalar")
    e = res.evm()

    metrics = _evm_metrics(e)
    f_trace = None
    if _is_bluetooth(wf):
        bt, f_trace = _bluetooth_metrics(res, wf)
        metrics.update(bt)
    else:
        try:
            a = res.aclr()
            metrics["ACLR upper [dBc]"] = round(float(a["upper_dbc"]), 1)
        except ValueError:
            pass
    f, pdb, mask_db, mask_metrics = _spectrum_metrics(res, wf)
    metrics.update(mask_metrics)

    cons = None
    if wf.kind == "ofdm":
        cons, ofdm_metrics = _ofdm_constellation(
            res, wf, e, eq, metrics.get("EVM [dB]", 0.0))
        metrics.update(ofdm_metrics)

    fig = _chain_figure(name, wf, res, e, f, pdb, mask_db, cons, f_trace)
    return {"metrics": metrics, "fig": fig, "result": res, "waveform": wf}


def save_setup(path: str, preset: str, overrides: dict | None = None, *,
               seed: int = 1, noise: bool = True) -> None:
    """Persist a workbench setup as JSON: (preset name, overrides,
    seed, noise) — the complete recipe run_chain_report needs.  Only
    JSON-serializable overrides are accepted (all GUI knobs are)."""
    import json
    doc = {"polartx_setup": 1, "preset": preset,
           "overrides": overrides or {}, "seed": seed, "noise": noise}
    try:                                # fail fast on non-serializable
        json.dumps(doc)
    except TypeError as exc:
        raise ValueError(f"overrides not JSON-serializable: {exc}") from exc
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)


def load_setup(path: str) -> dict:
    """Load a setup saved by save_setup; validates the preset name."""
    import json
    with open(path) as f:
        doc = json.load(f)
    if doc.get("polartx_setup") != 1:
        raise ValueError(f"{path} is not a polartx setup file")
    if doc["preset"] not in PRESETS:
        raise ValueError(f"unknown preset {doc['preset']!r}")
    doc.setdefault("overrides", {})
    doc.setdefault("seed", 1)
    doc.setdefault("noise", True)
    return doc


def run_setup(doc: dict) -> dict:
    """Execute a loaded setup through run_chain_report."""
    return run_chain_report(doc["preset"], seed=int(doc["seed"]),
                            noise=bool(doc["noise"]), **doc["overrides"])


# ------------------------------------------------------------ calibration
# The three calibration demos used to live inside the page code, written
# twice (Qt and Streamlit) and therefore untestable without a GUI; the
# pages now only lay these out.


def run_skew_cal(skew_ns: float = 2.0, *, method: str = "waveform align"
                 ) -> dict:
    """Inject an AM/PM skew on WiFi 160 MHz and estimate it back.

    ``method`` is ``"waveform align"`` (cross-correlate the delivered
    envelope against the commanded one) or ``"ACP power search"`` (the
    blind estimator that only needs the output spectrum)."""
    from .cal.skew import estimate_env_skew, estimate_skew_by_acp
    from .presets import wifi_dtc
    if method not in ("waveform align", "ACP power search"):
        raise ValueError(f"unknown skew estimator {method!r}")
    inj = float(skew_ns) * 1e-9
    p = wifi_dtc(bw=160e6, env_skew_s=inj)
    wf = p.make_waveform(n_symbols=4, seed=2)
    if method == "waveform align":
        est = estimate_env_skew(p.tx.run(wf, noise=False))
    else:
        est = estimate_skew_by_acp(p.tx, wf)
    return {"injected_ns": float(skew_ns),
            "estimated_ns": est["skew_s"] * 1e9,
            "error_ps": abs(est["skew_s"] - inj) * 1e12, "method": method}


def run_twopoint_cal(gain_err_pct: float = 3.0) -> dict:
    """Inject a direct-path gain error on LTE 20 MHz, estimate it offline
    (least squares on commanded vs delivered phase), correct, re-run."""
    from .cal.twopoint import estimate_dp_gain_error
    from .phasemod import ADPLLTwoPoint
    from .presets import lte20_adpll
    eps = float(gain_err_pct) / 100.0
    p = lte20_adpll(qam=64, dp_gain=1.0 + eps)
    pm = p.tx.phasemod
    assert isinstance(pm, ADPLLTwoPoint)
    wf = p.make_waveform(n_symbols=10, seed=0)
    r0 = p.tx.run(wf, noise=True, seed=2)
    est = estimate_dp_gain_error(pm, r0.phase_cmd, r0.phase_out, r0.fs)
    pm.dp_gain *= est["dp_gain_corr"]
    r1 = p.tx.run(wf, noise=True, seed=2)
    return {"injected_pct": float(gain_err_pct),
            "estimated_pct": 100.0 * est["eps_hat"],
            "evm_before_db": r0.evm().db, "evm_after_db": r1.evm().db}


def run_dtc_cal(iterations: int = 2) -> dict:
    """DTC gain/INL LUT calibration from a CW training tone: a 12-bit DTC
    with 1 % gain error and a 3rd-order sinusoidal INL, fitted and
    corrected ``iterations`` times."""
    from .cal.dtc_cal import apply_dtc_correction, fit_dtc_correction
    from .phasemod import DTCPhaseModulator, DTCPMConfig
    fs, n = 640e6, 1 << 16
    cw = 2 * np.pi * 5e6 * np.arange(n) / fs
    pm = DTCPhaseModulator(DTCPMConfig(
        n_bits=12, gain_error=0.01, inl_sin=(2e-3, 3, 0.0)))
    before = fit_dtc_correction(pm, cw, fs)["err_rms_before"]
    for _ in range(int(iterations)):
        apply_dtc_correction(pm, fit_dtc_correction(pm, cw, fs))
    after = fit_dtc_correction(pm, cw, fs)["err_rms_before"]
    return {"err_rms_before_mrad": before * 1e3,
            "err_rms_after_mrad": after * 1e3, "iterations": int(iterations)}


def run_fir_report(bw: float = 40e6, *, notch_offset_hz: float = 500e6,
                   n_symbols: int = 16, seed: int = 0, noise: bool = True,
                   osr: int = 50) -> dict:
    """Borokhovich RFIC'26 2-tap FIR + digital-Doherty benchmark.

    The preset is also a plain registry entry (the chain workbench scores
    it like any other).  This fuller view additionally runs the SINGLE-TAP
    baseline, which the headline OOC-noise suppression must be measured
    against — a lone dual-tap run cannot show what the notch bought.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .fir import ooc_noise_suppression_db
    from .presets import bench_wifi7_mlo_fir_borokhovich26

    p = bench_wifi7_mlo_fir_borokhovich26(bw=bw,
                                          notch_offset_hz=notch_offset_hz,
                                          osr=osr)
    wf = p.make_waveform(n_symbols=n_symbols, seed=seed)
    res = p.fir_tx.run(wf, noise=noise, seed=seed)
    base = p.single_tx.run(wf, noise=noise, seed=seed)

    # measure suppression in a band straddling the notch
    band = (0.9 * notch_offset_hz, 1.1 * notch_offset_hz)
    supp = ooc_noise_suppression_db(res, base, band)

    metrics = {"EVM FIR [dB]": round(res.evm().db, 1),
               "EVM single-tap [dB]": round(base.evm().db, 1),
               "notch offset [MHz]": round(notch_offset_hz / 1e6, 1),
               "tap delay tau [ps]": round(res.tau_s * 1e12, 1),
               "OOC suppression [dB]": round(supp, 1)}

    f1, p1 = base.psd(nfft=1 << 14)
    f2, p2 = res.psd(nfft=1 << 14)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(f1 / 1e6, p1, lw=0.6, alpha=0.75, label="single tap")
    ax[0].plot(f2 / 1e6, p2, lw=0.6, label="2-tap FIR")
    for s in (-1, 1):
        ax[0].axvline(s * notch_offset_hz / 1e6, color="r", ls=":", lw=1)
    ax[0].set(xlabel="offset [MHz]", ylabel="dBr", ylim=(-160, 5),
              title=f"OOC noise notch ({supp:.1f} dB at "
                    f"{notch_offset_hz/1e6:.0f} MHz)")
    ax[0].legend(fontsize=8)

    # the FIR magnitude response that produces the notch
    from .fir import fir_response
    fx = np.linspace(-1.5 * notch_offset_hz, 1.5 * notch_offset_hz, 2000)
    ax[1].plot(fx / 1e6, 20 * np.log10(np.abs(fir_response(fx, res.tau_s))
                                       + 1e-12))
    ax[1].set(xlabel="offset [MHz]", ylabel="|H| [dB]", ylim=(-60, 8),
              title="2-tap FIR response  H = 1 + exp(-j2*pi*f*tau)")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    return {"metrics": metrics, "fig": fig, "result": res, "baseline": base}


def run_selector_report(bw_hz: float = 80e6, *, standard: str = "custom",
                        modulation: str = "ofdm", evm_db_max: float = -35.0,
                        fout: float = 5.8e9, dtc_bits: int = 11,
                        two_point_gain_match: float = 2e-3,
                        constant_envelope: bool = False) -> dict:
    """Architecture selector: rank the four topologies (ADPLL two-point,
    open-loop DTC, outphasing, Cartesian RF-DAC) and chart the
    EVM-vs-bandwidth crossover around the requested point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .selector import Requirement, select

    req = Requirement(standard=standard, bw_hz=bw_hz, modulation=modulation,
                      evm_db_max=evm_db_max, fout=fout, dtc_bits=dtc_bits,
                      two_point_gain_match=two_point_gain_match,
                      constant_envelope=constant_envelope)
    rep = select(req)

    rows = {}
    for c in rep.candidates:
        rows[c.arch] = ("excluded" if not c.feasible
                        else f"{c.evm_db:.1f} dB, eta {c.eta_avg*100:.0f} %")
    rows["recommendation"] = rep.recommendation
    rows["closest preset"] = rep.suggest_preset()

    bws = np.logspace(6, np.log10(320e6), 40)
    curves: dict[str, list] = {"adpll_two_point": [], "dtc_open_loop": [],
                               "outphasing": [], "rfdac_cartesian": []}
    for b in bws:
        r = select(Requirement(standard="s", bw_hz=b, modulation=modulation,
                               evm_db_max=evm_db_max, fout=fout,
                               dtc_bits=dtc_bits,
                               two_point_gain_match=two_point_gain_match,
                               constant_envelope=constant_envelope))
        for c in r.candidates:
            curves[c.arch].append(c.evm_db if c.feasible else np.nan)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.semilogx(bws / 1e6, curves["adpll_two_point"], "-o", ms=3,
                label="ADPLL two-point")
    ax.semilogx(bws / 1e6, curves["dtc_open_loop"], "-^", ms=3,
                label="open-loop DTC")
    ax.semilogx(bws / 1e6, curves["outphasing"], "-s", ms=3,
                label="outphasing (2 DTC branches)")
    ax.semilogx(bws / 1e6, curves["rfdac_cartesian"], "-d", ms=3,
                label="Cartesian RF-DAC")
    ax.axvline(bw_hz / 1e6, color="k", ls=":", lw=1,
               label=f"request {bw_hz/1e6:.0f} MHz")
    ax.axhline(evm_db_max, color="r", ls="--", lw=1, label="EVM target")
    ax.set(xlabel="signal bandwidth [MHz]", ylabel="estimated EVM [dB]",
           title="architecture crossover (analytic)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return {"metrics": rows, "table": rep.table(), "fig": fig, "report": rep}


def run_combiner_report(n_way: int = 2, *, backoff_db: float = 6.0,
                        eta_peak: float = 0.85, peaking: str = "C",
                        combiner_loss_db: float = 0.4,
                        gain_imbalance_pct: float = 0.0,
                        phase_imbalance_deg: float = 0.0) -> dict:
    """Multi-core / Doherty combining: derived efficiency vs backoff and
    the AM-AM/AM-PM handoff distortion from core-to-core imbalance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .dpa.characteristics import efficiency_curve
    from .dpa.combiner import DohertyCombiner

    gi = tuple([0.0] + [gain_imbalance_pct / 100.0] * (n_way - 1))
    pi_ = tuple([0.0] + [phase_imbalance_deg] * (n_way - 1))
    c = DohertyCombiner(n_way=n_way, backoff_db=backoff_db,
                        eta_peak=eta_peak, peaking=peaking,
                        combiner_loss_db=combiner_loss_db,
                        gain_imbalance=gi, phase_imbalance_deg=pi_)
    cur = c.am_curves()

    x = np.linspace(0.02, 1.0, 400)
    eta_d = c.efficiency(x)
    eta_s = efficiency_curve(("scpa", 0.67, eta_peak), x)

    # average efficiency over an OFDM-like (Rayleigh) envelope
    rng = np.random.default_rng(0)
    env = np.abs(rng.normal(size=100_000) + 1j * rng.normal(size=100_000))
    env = np.clip(env / np.percentile(env, 99.9), 0, 1)
    p_out = env ** 2

    def _avg(eta_fn):
        e = eta_fn(env)
        on = e > 0
        return float(p_out[on].sum() / (p_out[on] / e[on]).sum())

    metrics = {
        "avg eff Doherty [%]": round(100 * _avg(c.efficiency), 1),
        "avg eff single-core SCPA [%]":
            round(100 * _avg(lambda v: efficiency_curve(
                ("scpa", 0.67, eta_peak), v)), 1),
        "peak eff [%]": round(100 * float(eta_d.max()), 1),
        "AM-AM ripple [dB]": round(cur["amam_ripple_db"], 3),
        "AM-PM pp [deg]": round(cur["ampm_pp_deg"], 2),
        "combining loss [dB]": round(c.combining_loss_db(), 2),
    }

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    bo = -20 * np.log10(x)
    ax[0].plot(bo, 100 * eta_d, label=f"{n_way}-way Doherty (class-{peaking})")
    ax[0].plot(bo, 100 * eta_s, "--", label="single-core SCPA")
    ax[0].axvline(backoff_db, color="k", ls=":", lw=1, label="backoff point")
    ax[0].set(xlabel="output backoff [dB]", ylabel="drain efficiency [%]",
              title="load-modulated efficiency")
    ax[0].set_xlim(18, 0)
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(fontsize=8)

    xs = np.clip(cur["x"], 1e-6, None)
    ax[1].plot(cur["x"], 20 * np.log10(np.clip(cur["amam"], 1e-6, None) / xs),
               label="AM-AM gain error [dB]")
    ax[1].plot(cur["x"], np.rad2deg(cur["ampm_rad"]), label="AM-PM [deg]")
    ax[1].axvline(10 ** (-backoff_db / 20), color="k", ls=":", lw=1,
                  label="handoff")
    ax[1].set(xlabel="input amplitude", title="core-imbalance distortion")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    return {"metrics": metrics, "fig": fig, "combiner": c}


def run_rtl_export(outdir: str, *, n_bits: int = 10, n_thermo: int = 7,
                   with_dpd: bool = True, verify: bool = True) -> dict:
    """Emit the digital polar-TX datapath + Verilog-AMS PA model and (if
    iverilog is installed) run every bit-true golden check."""
    from .cal.polar_dpd import PolarDPD
    from .dpa import DPA, DPAConfig
    from .export import rtl

    dpa = DPA(DPAConfig(n_bits=n_bits, n_thermo=n_thermo, sigma_cell=0.01,
                        amam=("rapp", 2.5, 1.1), ampm_deg_poly=(0.0, 2.0, 3.0)))
    dpd = PolarDPD.from_dpa(dpa) if with_dpd else None
    paths = rtl.emit_datapath(dpa, outdir, dpd=dpd)

    checks = {}
    if verify:
        for label, fn in (("CFR clip", rtl.verify_cfr_clip),
                          ("DTC phase acc", rtl.verify_phase_acc),
                          ("DPA thermo decoder", rtl.verify_thermo_decoder),
                          ("polar DPD LUT", rtl.verify_with_iverilog)):
            if label == "polar DPD LUT" and dpd is None:
                continue
            out = fn(outdir)
            checks[label] = ("iverilog not installed" if out is None else
                             next((ln for ln in out.splitlines()
                                   if "PASS" in ln or "FAIL" in ln), "?"))
    sc = rtl.dpa_rnm_selfcheck(dpa, outdir)
    checks["DPA RNM self-check"] = (
        f"{'OK' if sc['ok'] else 'FAIL'} (amp err {sc['amp_max_err']:.1e}, "
        f"AM-PM err {sc['ph_max_err']:.1e} rad)")
    return {"files": sorted(paths), "checks": checks, "outdir": outdir}


def run_mc_report(n_chips: int = 30, *, bw: float = 160e6,
                  skew_sigma_ns: float = 0.5, calibrated: bool = False,
                  limit_db: float = -35.0) -> dict:
    """Monte Carlo yield run rendered for the GUI: metrics, summary table
    and histogram figure.  Runs headless (Agg), so both frontends and the
    tests call the same function."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .montecarlo import run_mc, wifi_chip_builder

    mc = run_mc(wifi_chip_builder(bw=bw, n_symbols=3,
                                  skew_sigma_s=skew_sigma_ns * 1e-9,
                                  calibrated_skew=calibrated),
                n_chips, limit=limit_db)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(mc.values, bins=15, alpha=0.75)
    ax.axvline(limit_db, color="r", ls="--", label=f"limit {limit_db} dB")
    ax.set(xlabel="EVM [dB]", ylabel="chips",
           title=f"yield {100 * mc.yield_frac:.0f}% "
                 f"({'with' if calibrated else 'no'} skew cal)")
    ax.legend()
    fig.tight_layout()
    return {"summary": mc.summary(), "fig": fig}
