"""The single entry point the Android app calls.

Everything the phone can do goes through :func:`call`, which takes two
strings and returns a string.  That narrowness is deliberate:

* **It is testable without Android.**  ``tests/test_appbridge.py`` exercises
  every method the phone can reach, on a laptop, in milliseconds.  A bridge
  whose only test is "install the APK and tap around" is a bridge nobody
  tests, and this repository already learned that lesson on the web GUI.
* **One error path.**  A bad method name, bad arguments and a genuine bug in
  the library all come back as ``{"ok": false, "error": ..., "traceback":
  ...}``, so ``app.js`` needs one branch rather than a taxonomy.
* **Kotlin stays inert.**  Adding a feature is a function here plus a render
  in ``app.js`` — never a change to Java, Gradle or the emulator.

The methods below are deliberately thin.  All the computation already lives
in :mod:`polartx.guiutil`, which the Streamlit and PySide6 front ends also
call, so the phone is a third renderer of the same numbers rather than a
fourth implementation of them.  Logic that creeps in here is logic the
238-test suite does not cover.

**A method with no caller is half a feature.**  An entry in ``_METHODS``
that no page renders looks tested and does nothing.
``tests/test_android_parity.py`` fails in both directions from text alone —
no browser, no emulator.

What the phone deliberately does NOT get, and why (parity is a decision to
record, not a gap to discover later — see ``cairn/android-app.md``):

``run_rtl_export``
    Writes a directory of Verilog and verifies it by shelling out to
    ``iverilog``.  Neither the output nor the verifier means anything on a
    phone.
``run_mc_parallel``
    The report layer uses the serial ``run_mc``; a ``ProcessPoolExecutor``
    has no business forking under Android.
"""
from __future__ import annotations

import base64
import io
import json
import traceback
from typing import Any, Callable

import numpy as np

# --------------------------------------------------------------- helpers


def _clean(x: Any) -> Any:
    """Make a report JSON-serialisable.

    guiutil hands back numpy scalars, numpy bools and the occasional NaN;
    ``json.dumps`` raises on the first and emits bare ``NaN`` for the last,
    which is invalid JSON and fails in the WebView rather than here.
    """
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    return x


def _png(fig: Any, dpi: int = 130) -> str:
    """A matplotlib figure as a base64 PNG for an ``<img src>``.

    Note the absence of ``bbox_inches="tight"``: it silently changes the
    figure's extent, so any pixel coordinate sent alongside would describe a
    different image than the one on screen.  The figure is closed here or
    the phone leaks one per call.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _report(r: dict, *, metrics_key: str = "metrics") -> dict:
    """Strip a guiutil report down to what can cross the bridge.

    guiutil returns live objects beside the numbers (``result``,
    ``combiner``, ``report``, the Figure).  Only the metrics and a rendered
    PNG go to the page; passing objects across this boundary is what the
    one-function design exists to prevent.
    """
    out: dict[str, Any] = {metrics_key: _clean(r.get(metrics_key, {}))}
    if "table" in r:
        out["table"] = str(r["table"])
    if r.get("fig") is not None:
        out["png"] = _png(r["fig"])
    return out


# --------------------------------------------------------------- methods


def _version() -> dict:
    """Cheap round trip the page calls on boot to prove the bridge is live.

    It is also the call that pays the numpy/scipy/matplotlib import — several
    seconds on a phone — which is why the page shows a loading state until
    this returns rather than looking broken on launch.
    """
    import platform

    import polartx
    return {"lib": getattr(polartx, "__version__", "unknown"),
            "python": platform.python_version(),
            "numpy": np.__version__}


def _list_presets() -> dict:
    """Preset registry, split so the page can group benchmarks separately.

    Same list the two desktop GUIs drive, from ``guiutil.PRESETS`` — a new
    ``bench_*`` factory therefore reaches the phone automatically, and
    ``tests/test_benchmarks.py`` fails if one never reaches the registry.
    """
    from .guiutil import PRESETS
    names = list(PRESETS)
    return {"presets": [n for n in names if not n.startswith("Bench:")],
            "benchmarks": [n for n in names if n.startswith("Bench:")]}


def _chain(preset: str = "BLE LE-1M", seed: int = 1, noise: bool = True,
           n_units: int | None = None, env_skew_ns: float = 0.0,
           cfr_papr_db: float | None = None) -> dict:
    """Run one preset end to end: metrics table plus the report figure.

    ``env_skew_ns`` is exposed because AM/PM path skew is this
    architecture's sharpest impairment and the most instructive thing to
    drag a slider over — WiFi 160 MHz fails its mask at 0.2 ns, LTE-20 at
    1 ns, and constant-envelope BLE is immune.  It is given in ns rather
    than seconds so a phone slider has usable resolution.

    Measured on an x86 runner, every preset at its default burst length
    completes in 0.15-0.55 s; expect single-digit seconds on a phone.
    """
    from .guiutil import run_chain_report
    overrides: dict[str, Any] = {}
    if env_skew_ns:
        overrides["env_skew_s"] = float(env_skew_ns) * 1e-9
    if cfr_papr_db is not None:
        overrides["cfr_papr_db"] = float(cfr_papr_db)
    r = run_chain_report(preset, seed=int(seed), noise=bool(noise),
                         n_units=n_units, **overrides)
    return _report(r)


def _fir(bw: float = 40e6, notch_offset_hz: float = 500e6,
         n_symbols: int = 16, seed: int = 0, noise: bool = True,
         osr: int = 50) -> dict:
    """Dual-tap FIR + Doherty polar DTX (the RFIC'26 MLO benchmark).

    The notch has to fall inside the represented band or the suppression
    measurement averages an empty slice and returns NaN.  Refusing here is
    the house rule (a metric with insufficient input raises rather than
    returning NaN, which would reach the page as a blank card with nothing
    pointing at the cause).

    This is the slowest method on the phone: osr=50 is what puts a
    500 MHz notch inside the grid at all, and it measured 2.1 s on an x86
    runner.
    """
    nyquist = 0.5 * float(bw) * int(osr)
    if not 0 < float(notch_offset_hz) < nyquist:
        raise ValueError(
            f"notch offset {notch_offset_hz/1e6:.0f} MHz must fall inside "
            f"the represented band (0, {nyquist/1e6:.0f}) MHz — raise osr "
            f"or lower the offset; outside it the OOC suppression is NaN")
    from .guiutil import run_fir_report
    r = run_fir_report(bw=float(bw), notch_offset_hz=float(notch_offset_hz),
                       n_symbols=int(n_symbols), seed=int(seed),
                       noise=bool(noise), osr=int(osr))
    return _report(r)


def _selector(bw_hz: float = 80e6, standard: str = "custom",
              modulation: str = "ofdm", evm_db_max: float = -35.0,
              fout: float = 5.8e9, dtc_bits: int = 11,
              two_point_gain_match: float = 2e-3,
              constant_envelope: bool = False) -> dict:
    """Score both architectures against a requirement (analytic screen).

    Returns the verdict, the per-contributor breakdown table and the
    EVM-vs-bandwidth figure.  This is a screen, not a simulation: it says
    which architecture to reach for first, and ``chain`` is how you confirm
    it.
    """
    from .guiutil import run_selector_report
    r = run_selector_report(
        bw_hz=float(bw_hz), standard=str(standard), modulation=str(modulation),
        evm_db_max=float(evm_db_max), fout=float(fout),
        dtc_bits=int(dtc_bits),
        two_point_gain_match=float(two_point_gain_match),
        constant_envelope=bool(constant_envelope))
    return _report(r)


def _combiner(n_way: int = 2, backoff_db: float = 6.0,
              eta_peak: float = 0.85, peaking: str = "C",
              combiner_loss_db: float = 0.4,
              gain_imbalance_pct: float = 0.0,
              phase_imbalance_deg: float = 0.0) -> dict:
    """Multi-core / Doherty load-modulation combining.

    Efficiency here is DERIVED from the combining physics rather than
    fitted, which is why the class-B flat top and the class-C dip come out
    on their own instead of being drawn in.
    """
    from .guiutil import run_combiner_report
    r = run_combiner_report(
        n_way=int(n_way), backoff_db=float(backoff_db),
        eta_peak=float(eta_peak), peaking=str(peaking),
        combiner_loss_db=float(combiner_loss_db),
        gain_imbalance_pct=float(gain_imbalance_pct),
        phase_imbalance_deg=float(phase_imbalance_deg))
    return _report(r)


def _montecarlo(n_chips: int = 20, bw: float = 160e6,
                skew_sigma_ns: float = 0.5, calibrated: bool = False,
                limit_db: float = -35.0) -> dict:
    """Yield across a population of chips (per-chip mismatch/skew draws).

    Serial by construction: the report layer uses ``montecarlo.run_mc``, not
    the ``ProcessPoolExecutor`` variant, which has no business forking under
    Android.  Cost is linear in ``n_chips`` — 30 chips measured 0.6 s on an
    x86 runner, so keep the phone's default modest and let the user raise it.
    """
    from .guiutil import run_mc_report
    r = run_mc_report(n_chips=int(n_chips), bw=float(bw),
                      skew_sigma_ns=float(skew_sigma_ns),
                      calibrated=bool(calibrated), limit_db=float(limit_db))
    return _report(r, metrics_key="summary")


_METHODS: dict[str, Callable[..., Any]] = {
    "version": _version,
    "list_presets": _list_presets,
    "chain": _chain,
    "fir": _fir,
    "selector": _selector,
    "combiner": _combiner,
    "montecarlo": _montecarlo,
}


def call(method: str, args_json: str = "{}") -> str:
    """Single host entry point: dispatch, and never raise across the FFI.

    A Python exception crossing into Kotlin arrives as a PyException whose
    message the WebView cannot render usefully, so failures come back
    in-band.  The traceback is kept deliberately: it is for an engineer
    reading logcat, and debugging a phone without one is guessing.
    """
    try:
        fn = _METHODS.get(method)
        if fn is None:
            known = ", ".join(sorted(_METHODS))
            raise KeyError(f"no such method {method!r}; known: {known}")
        args = json.loads(args_json) if args_json else {}
        if not isinstance(args, dict):
            raise TypeError("arguments must be a JSON object")
        return json.dumps({"ok": True, "result": _clean(fn(**args))})
    except Exception as e:                       # noqa: BLE001 -- FFI boundary
        return json.dumps({"ok": False,
                           "error": f"{type(e).__name__}: {e}",
                           "traceback": traceback.format_exc()})
