"""Measured-data DPA modeling via the OpenDPD data path.

Loads real DPA input/output captures (vendored padpd loaders: OpenDPD
dataset folders, Cadence/MATLAB CSV), aligns them, and extracts the
static polar characteristics — binned AM-AM and AM-PM versus input
amplitude — into a polartx DPA model (LUT AM-AM + LUT AM-PM).  The
static-polar NMSE against the capture quantifies how much of the device
is code-static (what the polar DPD LUTs can fix) versus memory (what
needs the Cartesian ILA, cal.memory_dpd).

The memory itself is then inverted from the same capture as a RESIDUAL
model (``fit_residual_memory``): a vendored padpd ``PAModel`` trained on
the pair (static-model output, measured output), so it describes exactly
what the static polar model does not — and it plugs into
``PolarTX(memory=...)`` as the chain's post-DPA memory stage.

Scale convention — the one thing that must not drift
------------------------------------------------------
``PolarTX.run`` applies ``memory`` to ``y`` AFTER the DPA output has been
multiplied by ``fs_scale``, and the chain's nominal gain is one: with
``fs_scale_fixed = ch["fs_scale"]`` (the capture's input full scale) its
static output for the captured input is ``static_prediction(ch, x) /
ch["chain_gain"]`` — the measured device's gain divided out.  The
residual model is therefore trained on that chain-scale pair, not on the
measured volts, because its nonlinear terms are not scale invariant: fit
it one scale and run it at another and every |y|^k term is wrong.  Callers
MUST build the chain with ``ChainConfig(fs_scale_fixed=ch["fs_scale"])``
(and the waveform's peak at that full scale) for the model to see the
scale it was fitted at.

Alignment convention — integer only
------------------------------------
The capture is aligned to its input by the INTEGER bulk delay only.  A
residual sub-sample delay is linear memory (the device's group delay),
which the residual model's taps represent exactly; resampling it away
first injects a sinc interpolator the model then has to undo (on the
synthetic DUT: composite −46.6 dB with fractional resampling vs −52 dB
without, and the injected FIR taps come back within 2 %).  The fractional
lag is still estimated and reported in ``ch["align"]["lag_total"]``.

Slow state — the complete-source path
-------------------------------------
Thermal / bias / trapping memory acts on µs time scales: the device's gain
follows its recent envelope POWER, not the last few samples.  A stationary
capture cannot see it (the state is constant), so it is inverted from the
padpd "complete source" container (``vendor.padpd.data.complete``) instead:
its ``step`` group (a constant-envelope step probe) identifies the time
constants offline, and its ``burst`` group (power-stepped OFDM, heating
and cooling) trains a ``StateConditionedSpline`` residual whose slow
power states use exactly those constants.  ``fit_residual_memory(ch,
source=...)`` does both; ``dpa_from_complete_source`` is the one-call
form.  The identified time constants are AM->AM / AM->PM gain-modulation
constants of the PA.  They are NOT the supply network's ``SupplyConfig.
tau_s`` (an AM->PM ripple filter) and must not be copied into it.

The state model recomputes its slow states causally from the start of
every call, i.e. a chain run is a COLD start — exactly what the DUT in a
burst capture is.  Pinning the scale (``fs_scale_fixed``) matters even
more here than for the fast residual: the states are powers, so a scale
error enters squared.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

from .dpa import DPA, DPAConfig
from .vendor.padpd.data import (align_delay, load_complete_npz,
                                load_opendpd_dataset, save_complete_npz)
from .vendor.padpd.gain_modulation import (GainModulationResult,
                                           identify_gain_modulation_capture,
                                           step_probe_drive)
from .vendor.padpd.pa import (GMPModel, PAModel, SplineGMP,
                              StateConditionedSpline, gmp_opendpd_510,
                              nmse_db)
from .vendor.padpd.pa.thermal import ThermalReferencePA, burst_stimulus
from .vendor.padpd.waveform.ofdm import OFDMConfig, generate_ofdm

OPENDPD_ENV = "POLARTX_OPENDPD"

#: complete-source groups the slow-state fit needs, and what each is for
SLOW_STATE_GROUPS = {
    "step": "a constant-envelope step probe (padpd.gain_modulation."
            "step_probe_drive) -> the gain-modulation time constants",
    "burst": "a power-stepped capture (heating AND cooling) -> the "
             "state-conditioned residual's training data",
}


def find_opendpd_root() -> str | None:
    """Locate an OpenDPD clone: $POLARTX_OPENDPD, then ../OpenDPD."""
    for cand in (os.environ.get(OPENDPD_ENV),
                 os.path.join(os.path.dirname(__file__),
                              "..", "..", "..", "OpenDPD"),
                 "../OpenDPD"):
        if cand and os.path.isdir(os.path.join(cand, "datasets")):
            return os.path.abspath(cand)
    return None


def _align(x: np.ndarray, y: np.ndarray, fractional: bool) -> tuple:
    """Bulk-align ``y`` to ``x``; see the module docstring for why the
    default keeps the sub-sample part in the data."""
    x_a, y_a, info = align_delay(x, y)
    if fractional:
        return x_a, y_a, dict(info, fractional_removed=True)
    lag = int(info["lag"])
    if lag >= 0:
        xs, ys = x[:x.size - lag] if lag else x, y[lag:]
    else:
        xs, ys = x[-lag:], y[:y.size + lag]
    n = min(xs.size, ys.size)
    return xs[:n], ys[:n], dict(info, fractional_removed=False)


def extract_polar_characteristics(x: np.ndarray, y: np.ndarray, *,
                                  n_bins: int = 64,
                                  fractional_align: bool = False) -> dict:
    """Aligned, binned static polar model of a measured PA/DPA.

    Returns normalized AM-AM (r_in [0,1] -> r_out [0,1]), AM-PM [deg]
    versus r_in, the linear gain, and the static-polar NMSE: how well
    y is explained by g * lut(|x|) * exp(j(angle(x) + ampm(|x|))) —
    the same ``static_prediction`` the residual model is trained
    against.  Also carried, for the chain: ``fs_scale`` (the input
    full scale, ``max|x|``) and ``chain_gain`` (what divides the
    measured output into the chain's unity-gain scale).
    """
    x_a, y_a, info = _align(np.asarray(x, complex), np.asarray(y, complex),
                            fractional_align)
    r = np.abs(x_a)
    r_max = float(r.max())
    rn = r / r_max
    ratio = y_a / np.where(np.abs(x_a) > 1e-12 * r_max, x_a, np.nan)
    gain = np.abs(ratio)
    phase = np.angle(ratio)
    # the absolute rotation is free: the chain's DPA has no absolute
    # phase, so the residual model's linear term carries it (−0.4 deg on
    # the OpenDPD captures)
    phase = phase - np.nanmedian(phase)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(rn, edges) - 1, 0, n_bins - 1)
    g_bin = np.full(n_bins, np.nan)
    p_bin = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = (idx == b) & np.isfinite(gain)
        if m.sum() >= 8:
            g_bin[b] = np.nanmean(gain[m])
            p_bin[b] = np.nanmean(phase[m])
    centers = 0.5 * (edges[:-1] + edges[1:])
    ok = np.isfinite(g_bin)
    centers, g_bin, p_bin = centers[ok], g_bin[ok], p_bin[ok]

    r_out = centers * g_bin
    r_out = r_out / r_out[-1]
    ampm_deg = np.rad2deg(p_bin)

    ch = {"r_in": centers, "r_out": r_out, "ampm_deg": ampm_deg,
          "gain": float(np.nanmedian(gain)),
          "align": info, "x_aligned": x_a, "y_aligned": y_a,
          # the static model's own tables, so static_prediction is the
          # same code path the NMSE below is measured with
          "_g_bin": g_bin, "_p_bin": p_bin, "_r_max": r_max,
          # input full scale and the measured-to-chain gain: the chain's
          # DPA LUT is r_out normalized to 1 at its top bin, so for the
          # captured input it outputs static_prediction / chain_gain
          "fs_scale": r_max,
          "chain_gain": float(centers[-1] * g_bin[-1])}
    ch["static_nmse_db"] = nmse_db(y_a, static_prediction(ch, x_a))
    return ch


def static_prediction(ch: dict, x: np.ndarray) -> np.ndarray:
    """The static polar model's output for input ``x``, in the MEASURED
    output scale: ``g(|x|) · x · exp(j·ampm(|x|))`` with the binned
    tables of ``extract_polar_characteristics``.  This is exactly what
    ``static_nmse_db`` scores and what the residual model is trained on
    (divided by ``ch["chain_gain"]``)."""
    x = np.asarray(x, complex)
    rn = np.abs(x) / ch["_r_max"]
    g_of_r = np.interp(rn, ch["r_in"], ch["_g_bin"])
    p_of_r = np.interp(rn, ch["r_in"], ch["_p_bin"])
    return x * g_of_r * np.exp(1j * p_of_r)


def residual_training_pair(ch: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(u, v)`` in CHAIN scale: the static model's output and the
    measured output for the aligned capture, both divided by
    ``ch["chain_gain"]`` so that ``PolarTX(memory=M)`` built with
    ``fs_scale_fixed=ch["fs_scale"]`` sees ``u`` and should produce ``v``."""
    g = ch["chain_gain"]
    return (static_prediction(ch, ch["x_aligned"]) / g, ch["y_aligned"] / g)


def memoryless_residual_factory(order: int = 5) -> GMPModel:
    """A GMP with memory_depth 1 and no cross terms: a static polynomial
    gain correction.  Its NMSE gain over the static LUT is the part of
    the static-vs-measured gap that is NOT memory — the control that
    decides whether '19 dB is memory' is a fair reading."""
    return GMPModel(order=order, memory_depth=1, lag_order=0, lag_memory=0,
                    lag_span=0, lead_order=0, lead_memory=0, lead_span=0)


def spline_gmp_residual(u_train: np.ndarray, n_knots: int = 3,
                        degree: int = 3) -> SplineGMP:
    """SplineGMP on GMP-510's memory structure (15 / 15x2 / 15x1) with
    knots placed on the training segment's amplitude quantiles: at
    ``n_knots=3, degree=3`` it has the same 255 coefficients as GMP-510.
    Measured on DPA_160MHz: NMSE within 0.1 dB of GMP-510, condition
    number ~1.5 orders of magnitude lower (not the 2-3 orders the direct
    x->y fits in PA_DPD show — the residual input is already a compressed
    envelope, which flattens the polynomial basis's collinearity)."""
    return SplineGMP.from_signal(u_train, n_knots=n_knots, degree=degree,
                                 memory_depth=15, lag_memory=15, lag_span=2,
                                 lead_memory=15, lead_span=1,
                                 placement="quantile")


def linear_taps(model: PAModel) -> np.ndarray:
    """The first-order (``x·|x|^0``) coefficient of each aligned memory
    tap of a GMP / memory-polynomial model: the model's linear FIR."""
    order = getattr(model, "order", None)
    depth = getattr(model, "memory_depth", None)
    coeffs = getattr(model, "coeffs", None)
    if order is None or depth is None or coeffs is None:
        raise TypeError("linear_taps needs a fitted polynomial model with "
                        "order and memory_depth")
    return np.asarray(coeffs)[:order * depth][0::order]


def fast_spline_residual(u_train: np.ndarray) -> SplineGMP:
    """The fast-memory-only spline baseline the slow-state model is judged
    against: SplineGMP, 8 knots, 4 aligned taps, one lag and one lead
    cross branch (PA_DPD's comparison structure)."""
    return SplineGMP.from_signal(u_train, n_knots=8, memory_depth=4,
                                 lag_memory=2, lag_span=1,
                                 lead_memory=2, lead_span=1)


def state_spline_residual(u_train: np.ndarray,
                          state_alphas) -> StateConditionedSpline:
    """SMP (8 knots, 4 taps) plus slow envelope-power states with the
    given per-sample smoothing factors (``exp(-1/(tau*fs))``), spline-
    expanded additively and as a tap-0 amplitude x state surface."""
    return StateConditionedSpline.from_signal(u_train, n_knots=8,
                                              memory_depth=4,
                                              state_alphas=state_alphas)


def _require_groups(source: dict, groups) -> None:
    present = set((source.get("extras") or {}))
    missing = [g for g in groups if g not in present]
    if missing:
        detail = "; ".join(f"'{g}' = {SLOW_STATE_GROUPS[g]}" for g in missing)
        raise ValueError(
            "the slow-state fit needs the complete-source capture group(s) "
            f"{', '.join(repr(g) for g in missing)}, missing from this "
            f"source ({detail}).  A stationary capture cannot reveal slow "
            "state; record them (padpd.data.complete documents the format).")


def identify_slow_states(source: dict) -> GainModulationResult:
    """Offline time-constant identification from the source's ``step``
    capture.  Raises if the group is missing or the probe shows no
    significant gain modulation (then there is no slow state to model)."""
    _require_groups(source, ("step",))
    st = source["extras"]["step"]
    res = identify_gain_modulation_capture(st["x"], st["y"], source["fs"])
    if not res.significant:
        raise ValueError("the step capture shows no significant gain "
                         "modulation: no slow state to model")
    return res


def _basis_health(model, x: np.ndarray, n_samples: int = 8192
                  ) -> tuple[float, int, int]:
    """``(condition number on the spanned column space, rank deficiency,
    identically-zero columns)`` of the design matrix on ``x[:n_samples]``.

    For a full-rank basis (every GMP / spline-GMP fit here) the first is
    exactly ``basis_cond`` and the other two are 0.  A
    ``StateConditionedSpline`` is rank deficient BY CONSTRUCTION: each
    additive state block ``x[n-m]·C_i(q)`` sums over ``i`` to ``x[n-m]``
    (partition of unity), which its baseline tap already spans, and the
    interaction surface repeats the tap-0 columns the same way — upstream
    relies on the ridge term to pick the minimum-norm solution (SplineGMP
    drops one column per cross branch instead).  On top of that its one
    ``q_scale`` (the peak of the FASTEST state) keeps the slowest state
    off its top knots, leaving columns all zero.  The raw condition number
    therefore reads ~1e18; the ratio of the largest to the smallest
    singular value the data actually excites is the number that says
    whether the fit is healthy, and the deficiency is reported with it so
    nothing is hidden."""
    phi = model.basis_matrix(np.asarray(x)[:n_samples])
    dead = int((np.abs(phi).sum(axis=0) == 0).sum())
    s = np.linalg.svd(phi, compute_uv=False)
    tol = s[0] * max(phi.shape) * np.finfo(float).eps
    rank = int((s > tol).sum())
    return float(s[0] / s[rank - 1]), int(phi.shape[1] - rank), dead


def _fit_on_pair(u: np.ndarray, v: np.ndarray, model_factory, split: float,
                 regularization: float, ch: dict) -> tuple[Any, dict]:
    """Time-ordered train/validate fit of one residual model on (u, v)."""
    if not 0.0 < split < 1.0:
        raise ValueError("split must be in (0, 1)")
    n_tr = int(split * u.size)
    val = slice(n_tr, None)
    u_tr, v_tr = u[:n_tr], v[:n_tr]

    # Any: every vendored coefficient model takes fit(..., regularization=)
    # and exposes .coeffs; the abstract PAModel does not promise either
    model: Any = model_factory(u_tr)
    model.fit(u_tr, v_tr, regularization=regularization)
    # full-capture prediction scored on the tail: taps warm up and slow
    # states carry the same power history the device had
    y_full = model(u)

    ml = memoryless_residual_factory()
    ml.fit(u_tr, v_tr, regularization=regularization)
    cond, deficiency, dead = _basis_health(model, u_tr)

    report = {
        "split": split, "n_train": n_tr, "n_val": u.size - n_tr,
        "static_nmse_db": nmse_db(v[val], u[val]),
        "memoryless_nmse_db": nmse_db(v[val], ml(u)[val]),
        "residual_nmse_db": nmse_db(v[val], y_full[val]),
        "residual_train_nmse_db": nmse_db(v_tr, y_full[:n_tr]),
        "n_coeffs": int(np.asarray(model.coeffs).size),
        "condition_number": cond,
        "rank_deficiency": deficiency,
        "dead_columns": dead,
        "model_class": type(model).__name__,
        "chain_gain": ch["chain_gain"], "fs_scale": ch["fs_scale"],
    }
    return model, report


def fit_residual_memory(ch: dict,
                        model_factory: Callable[[], PAModel] | None = None,
                        *, split: float | None = None,
                        regularization: float = 1e-9,
                        source: dict | None = None) -> tuple[PAModel, dict]:
    """Fit the residual memory model.

    Without ``source`` — FAST memory from the capture in ``ch``: training
    pair ``residual_training_pair(ch)`` (chain scale), the first ``split``
    (default 0.8) of the capture IN TIME ORDER trains and the rest
    validates — no shuffling, so the validation set is a genuinely unseen
    stretch and the memory taps warm up inside the training part.
    ``model_factory`` defaults to GMP-510 (``gmp_opendpd_510``), the
    structure OpenDPD publishes its −39 dB with.

    With ``source`` (a ``load_complete_npz`` dict) — SLOW state: the
    time constants come from its ``step`` group (``identify_slow_states``),
    the training pair is its ``burst`` group passed through the static
    model in ``ch`` (both chain scale), and ``model_factory`` defaults to
    ``state_spline_residual`` with the identified constants.  ``split``
    defaults to 0.6 here: with four bursts it leaves 3.2 of the 8 power
    segments — both heating and cooling edges — in the validation tail
    (0.8 would leave 1.6).  Missing groups raise and say which.  The
    report adds ``taus_heat_s`` / ``taus_cool_s`` / ``state_alphas``,
    ``fast_only_nmse_db`` (the same pair fitted by
    ``fast_spline_residual``, no states) and ``state_gain_db``, the slow
    state's own contribution.

    Every report separates, on the validation segment: ``static_nmse_db``
    (the LUT alone), ``memoryless_nmse_db`` (LUT + a static 5th-order
    polynomial correction: what a finer static model could still buy)
    and ``residual_nmse_db`` (the full model), plus ``n_coeffs``,
    ``condition_number`` (on the column space the training input spans;
    ``rank_deficiency`` / ``dead_columns`` say what it excludes) and the
    training-set NMSE for over-fit inspection.
    """
    if source is None:
        def make(_u):
            return model_factory() if model_factory is not None else gmp_opendpd_510()
        return _fit_on_pair(*residual_training_pair(ch), make,
                            0.8 if split is None else split,
                            regularization, ch)

    _require_groups(source, ("step", "burst"))
    gm = identify_slow_states(source)
    alphas = gm.state_alphas(source["fs"])
    b = source["extras"]["burst"]
    xb, yb, _ = _align(np.asarray(b["x"], complex),
                       np.asarray(b["y"], complex), False)
    g = ch["chain_gain"]
    u, v = static_prediction(ch, xb) / g, yb / g
    split = 0.6 if split is None else split

    def make_state(u_tr):
        if model_factory is not None:
            return model_factory()
        return state_spline_residual(u_tr, alphas)

    model, report = _fit_on_pair(u, v, make_state, split, regularization, ch)
    _, fast = _fit_on_pair(u, v, fast_spline_residual, split,
                           regularization, ch)
    report.update({
        "source": "complete:burst",
        "taus_heat_s": list(gm.taus_heat_s),
        "taus_cool_s": list(gm.taus_cool_s),
        "state_alphas": list(alphas),
        "fast_only_nmse_db": fast["residual_nmse_db"],
        "fast_only_model_class": fast["model_class"],
        "state_gain_db": fast["residual_nmse_db"] - report["residual_nmse_db"],
    })
    return model, report


def dpa_from_measured(x: np.ndarray, y: np.ndarray, *, n_bits: int = 10,
                      n_bins: int = 64, **dpa_kw) -> tuple[DPA, dict]:
    """Build a polartx DPA whose code tables follow the measured device."""
    ch = extract_polar_characteristics(x, y, n_bins=n_bins)
    cfg = DPAConfig(n_bits=n_bits,
                    amam=("lut", ch["r_in"], ch["r_out"]),
                    ampm_lut=(ch["r_in"], ch["ampm_deg"]), **dpa_kw)
    return DPA(cfg), ch


def load_measured_dpa(dataset: str = "DPA_160MHz", *, split: str = "train",
                      n_bits: int = 10, root: str | None = None,
                      with_memory: bool = False,
                      model_factory: Callable[[], PAModel] | None = None
                      ) -> tuple[DPA, dict]:
    """OpenDPD dataset folder -> polartx DPA model + extraction report.

    With ``with_memory=True`` the report also carries ``ch["memory"]``,
    the residual memory model fitted by ``fit_residual_memory``, and
    ``ch["memory_report"]``.  Use it as::

        dpa, ch = load_measured_dpa("DPA_160MHz", with_memory=True)
        tx = PolarTX(ChainConfig(fs_scale_fixed=ch["fs_scale"]),
                     IdealPhaseModulator(), dpa, memory=ch["memory"])

    ``fs_scale_fixed=ch["fs_scale"]`` is not optional: the residual model
    was fitted on the chain's post-``fs_scale`` output scale for the
    captured drive level (module docstring), and the chain must run at
    the capture's sample rate ``ch["fs"]`` because the model's taps are
    in samples.
    """
    root = root or find_opendpd_root()
    if root is None:
        raise FileNotFoundError(
            f"OpenDPD clone not found; set ${OPENDPD_ENV} or clone next "
            "to the repo (git clone --depth 1 "
            "https://github.com/lab-emi/OpenDPD.git)")
    d = load_opendpd_dataset(os.path.join(root, "datasets", dataset))
    ds = d[split]
    dpa, ch = dpa_from_measured(ds.x, ds.y, n_bits=n_bits)
    ch["spec"] = d["spec"]
    ch["fs"] = ds.sample_rate_hz
    if with_memory:
        ch["memory"], ch["memory_report"] = fit_residual_memory(
            ch, model_factory)
    return dpa, ch


def dpa_from_complete_source(source: dict | str, *, n_bits: int = 10,
                             with_memory: bool = True,
                             model_factory: Callable[[], PAModel] | None = None
                             ) -> tuple[DPA, dict]:
    """Complete measured source (``load_complete_npz`` dict or a path) ->
    polartx DPA + report, with the SLOW-state residual memory.

    The static polar model comes from the source's main ``(x, y)``
    capture, which should be stationary in the thermal sense (recorded
    warm, and several of the slowest time constants long); the memory
    from ``fit_residual_memory(ch, source=...)``.  Chain use is the same
    as ``load_measured_dpa``: ``fs_scale_fixed=ch["fs_scale"]``,
    ``wf.fs == ch["fs"]``, and a run is a cold start.
    """
    src = load_complete_npz(source) if isinstance(source, str) else source
    dpa, ch = dpa_from_measured(src["x"], src["y"], n_bits=n_bits)
    ch["fs"] = src["fs"]
    ch["source_meta"] = dict(src.get("meta") or {})
    if with_memory:
        ch["memory"], ch["memory_report"] = fit_residual_memory(
            ch, model_factory, source=src)
    return dpa, ch


#: ground truth of ``synthetic_thermal_source``'s virtual DUT
SYNTHETIC_THERMAL_TAUS_S = (5e-6, 3e-5)


def synthetic_thermal_source(path: str | None = None, *, fs: float = 80e6,
                             warm_reps: int = 2) -> dict:
    """A complete measured source recorded from a VIRTUAL self-heating DUT
    (vendored ``ThermalReferencePA``: Wiener-Hammerstein Saleh PA whose
    drift state follows its own output power through a two-pole thermal
    network, taus ``SYNTHETIC_THERMAL_TAUS_S``) — the stage that lets the
    slow-state path run and be tested with no hardware.

    Groups, each from a fresh DUT instance:

    - main ``(x, y)``: 16 OFDM symbols (218 µs at 80 MS/s, ~7x the slowest
      tau) recorded after ``warm_reps`` passes of the same waveform, i.e.
      in thermal steady state — a STATIONARY capture.  Recorded cold or
      short it is a heating transient instead and the "stationary"
      control below is meaningless (measured: a cold 54 µs capture makes
      the state model look 5.4 dB better than a stateless one);
    - ``burst``: 16 symbols, 4 full-power / 0.3x bursts, cold start;
    - ``step``: ``step_probe_drive`` after a full-power reference burst
      (PA_DPD's recipe: the DUT calibrates its dissipation reference on
      its first call).

    With ``path`` the source is written with ``save_complete_npz`` and
    read back, so the result has been through the file format
    (complex64 storage) exactly as a recorded one would.
    """
    def ofdm(n_symbols, seed):
        return generate_ofdm(OFDMConfig(bandwidth_hz=fs / 4, qam_order=1024,
                                        n_symbols=n_symbols, seed=seed)).x

    def dut():
        return ThermalReferencePA(drive0=0.13, taus_s=SYNTHETIC_THERMAL_TAUS_S,
                                  fs=fs)

    x = ofdm(16, 0)
    pa_main = dut()
    for _ in range(warm_reps):
        pa_main(x)
    y = pa_main(x)

    xb = burst_stimulus(ofdm(16, 3), n_bursts=4, low_scale=0.3)
    yb = dut()(xb)

    xs = step_probe_drive(fs, settle_factor=4.0)
    pa_step = dut()
    pa_step(np.full(2048, 1.5, dtype=complex))
    pa_step.reset()
    ys = pa_step(xs)

    meta = {"reference_plane": "pa_output", "virtual_dut": "ThermalReferencePA",
            "true_taus_s": list(SYNTHETIC_THERMAL_TAUS_S),
            "main_capture": f"stationary: recorded after {warm_reps} warm-up passes"}
    if path is not None:
        save_complete_npz(path, x, y, fs, burst=(xb, yb), step=(xs, ys),
                          meta=meta)
        return load_complete_npz(path)
    return {"x": x, "y": y, "fs": float(fs), "meta": meta,
            "extras": {"burst": {"x": xb, "y": yb},
                       "step": {"x": xs, "y": ys}}}
