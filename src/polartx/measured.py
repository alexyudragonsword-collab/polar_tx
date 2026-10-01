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
"""
from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

from .dpa import DPA, DPAConfig
from .vendor.padpd.data import align_delay, load_opendpd_dataset
from .vendor.padpd.pa import (GMPModel, PAModel, SplineGMP, basis_cond,
                              gmp_opendpd_510, nmse_db)

OPENDPD_ENV = "POLARTX_OPENDPD"


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


def fit_residual_memory(ch: dict,
                        model_factory: Callable[[], PAModel] | None = None,
                        *, split: float = 0.8,
                        regularization: float = 1e-9) -> tuple[PAModel, dict]:
    """Fit the residual memory model on the capture in ``ch``.

    Training pair: ``residual_training_pair(ch)`` (chain scale).  The
    first ``split`` of the capture IN TIME ORDER trains, the rest
    validates — no shuffling, so the validation set is a genuinely unseen
    stretch of the burst and the memory taps warm up inside the training
    part.  ``model_factory`` defaults to GMP-510 (``gmp_opendpd_510``),
    the structure OpenDPD publishes its −39 dB with.

    The report separates the three numbers the comparison rests on, all
    on the validation segment: ``static_nmse_db`` (the LUT alone),
    ``memoryless_nmse_db`` (LUT + a static 5th-order polynomial
    correction: what a finer static model could still buy) and
    ``residual_nmse_db`` (the full model), plus ``n_coeffs``,
    ``condition_number`` (``basis_cond`` on the training input) and the
    training-set NMSE for over-fit inspection.
    """
    if not 0.0 < split < 1.0:
        raise ValueError("split must be in (0, 1)")
    u, v = residual_training_pair(ch)
    n_tr = int(split * u.size)
    val = slice(n_tr, None)
    u_tr, v_tr = u[:n_tr], v[:n_tr]

    # Any: every vendored coefficient model takes fit(..., regularization=)
    # and exposes .coeffs; the abstract PAModel does not promise either
    model: Any = model_factory() if model_factory is not None else gmp_opendpd_510()
    model.fit(u_tr, v_tr, regularization=regularization)
    y_full = model(u)                      # warm-up inside the training part

    ml = memoryless_residual_factory()
    ml.fit(u_tr, v_tr, regularization=regularization)

    report = {
        "split": split, "n_train": n_tr, "n_val": u.size - n_tr,
        "static_nmse_db": nmse_db(v[val], u[val]),
        "memoryless_nmse_db": nmse_db(v[val], ml(u)[val]),
        "residual_nmse_db": nmse_db(v[val], y_full[val]),
        "residual_train_nmse_db": nmse_db(v_tr, y_full[:n_tr]),
        "n_coeffs": int(np.asarray(model.coeffs).size),
        "condition_number": basis_cond(model, u_tr),
        "model_class": type(model).__name__,
        "chain_gain": ch["chain_gain"], "fs_scale": ch["fs_scale"],
    }
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
