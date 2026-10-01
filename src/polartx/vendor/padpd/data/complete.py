# Vendored from PA_DPD@44cbcb3: src/padpd/data/complete.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Complete measured-source container: one file, every capture the
platform can consume.

A plain (x, y, fs) capture feeds forward modeling and ILA DPD — nothing
else. Each additional, cheap capture group unlocks one capability that
is otherwise blind:

======================  =================================================
capture group           unlocks
======================  =================================================
``burst``               slow-state modeling (StateConditionedSpline) —
                        thermal states are unobservable in a stationary
                        capture
``step``                constant-envelope step-probe capture ->
                        gain-modulation time constants (tau ->
                        state_alphas), identified OFFLINE from the
                        recording
``cal_rx``              PA-bypass capture (known signal through the
                        observation receiver only) -> RX widely-linear +
                        inverse-FIR de-embedding calibration
``atten``               two captures of the same drive at a known RX
                        attenuator step -> RX-IM3 kappa calibration
``operating_points``    one (x, y) pair per operating condition
                        (temperature / bias / supply scalar) ->
                        CoefficientScheduler
======================  =================================================

File format: a single ``.npz`` with the arrays below. ``x, y,
sample_rate_hz`` are required; every group is optional and validated
only when present. Arrays are stored as complex64/float32 to keep files
small; loading restores complex128.

Required:            ``x``, ``y``, ``sample_rate_hz``
burst:               ``burst_x``, ``burst_y``
step:                ``step_x``, ``step_y``
cal_rx:              ``cal_rx_ref``, ``cal_rx_obs``
atten:               ``atten_ref``, ``atten_hi``, ``atten_lo``,
                     ``atten_step_db``
operating_points:    ``op_conditions`` (float array, one scalar per
                     point), ``op0_x``, ``op0_y``, ``op1_x``, ...
meta:                repr'd dict (center frequency, reference plane,
                     absolute power calibration, shared-LO flag, ...)

A regular :class:`~padpd.data.dataset.IQDataset` npz is a valid
degenerate case (no groups); conversely this file loads fine through
``IQDataset.load`` (extras simply ignored).
"""

from __future__ import annotations

import ast

import numpy as np

GROUPS = ("burst", "step", "cal_rx", "atten", "operating_points")

# capability text per group, used by GUIs/tools for the checklist
GROUP_UNLOCKS = {
    "burst": "StateConditionedSpline",
    "step": "identify_gain_modulation (tau -> state_alphas)",
    "cal_rx": "calibrate_rx_path (RX WL + inverse FIR)",
    "atten": "calibrate_rx_im3 (RX cubic kappa)",
    "operating_points": "CoefficientScheduler",
}


def _c64(v) -> np.ndarray:
    a = np.asarray(v)
    if a.ndim != 1 or len(a) == 0:
        raise ValueError("capture arrays must be non-empty 1-D")
    return a.astype(np.complex64)


def save_complete_npz(path: str, x, y, sample_rate_hz: float, *,
                      burst: tuple | None = None,
                      step: tuple | None = None,
                      cal_rx: tuple | None = None,
                      atten: tuple | None = None,
                      operating_points: tuple | None = None,
                      meta: dict | None = None) -> None:
    """Write a complete measured source.

    ``burst``/``step``/``cal_rx`` are (ref, obs) array pairs; ``atten``
    is (ref, obs_hi, obs_lo, step_db); ``operating_points`` is
    (conditions, [(x0, y0), (x1, y1), ...]) with one scalar condition
    per pair.
    """
    x = _c64(x)
    y = _c64(y)
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    arrays: dict = {"x": x, "y": y,
                    "sample_rate_hz": float(sample_rate_hz),
                    "meta": np.array(repr(meta or {}))}
    for key, pair in (("burst", burst), ("step", step)):
        if pair is not None:
            a, b = pair
            arrays[f"{key}_x"] = _c64(a)
            arrays[f"{key}_y"] = _c64(b)
            if len(arrays[f"{key}_x"]) != len(arrays[f"{key}_y"]):
                raise ValueError(f"{key} capture pair length mismatch")
    if cal_rx is not None:
        ref, obs = cal_rx
        arrays["cal_rx_ref"] = _c64(ref)
        arrays["cal_rx_obs"] = _c64(obs)
    if atten is not None:
        ref, hi, lo, step_db = atten
        arrays["atten_ref"] = _c64(ref)
        arrays["atten_hi"] = _c64(hi)
        arrays["atten_lo"] = _c64(lo)
        arrays["atten_step_db"] = float(step_db)
    if operating_points is not None:
        conditions, pairs = operating_points
        conditions = np.asarray(conditions, dtype=np.float64)
        if len(conditions) != len(pairs):
            raise ValueError("one condition scalar per operating point")
        if len(pairs) < 2:
            raise ValueError("need at least 2 operating points")
        arrays["op_conditions"] = conditions
        for i, (xi, yi) in enumerate(pairs):
            arrays[f"op{i}_x"] = _c64(xi)
            arrays[f"op{i}_y"] = _c64(yi)
    np.savez_compressed(path, **arrays)


def load_complete_npz(path: str) -> dict:
    """Load a complete (or plain) measured-source npz.

    Returns ``{"x", "y", "fs", "meta", "extras"}`` where ``extras`` maps
    each PRESENT group name to its arrays (complex128), e.g.
    ``extras["atten"] = {"ref", "hi", "lo", "step_db"}``. Groups that
    are absent simply do not appear.
    """
    d = np.load(path, allow_pickle=False)
    for req in ("x", "y", "sample_rate_hz"):
        if req not in d:
            raise ValueError(f"{path}: missing required array '{req}'")
    out = {"x": np.asarray(d["x"], dtype=complex),
           "y": np.asarray(d["y"], dtype=complex),
           "fs": float(d["sample_rate_hz"]),
           "meta": (ast.literal_eval(str(d["meta"]))
                    if "meta" in d else {}),
           "extras": {}}
    ex = out["extras"]
    for key in ("burst", "step"):
        if f"{key}_x" in d:
            ex[key] = {"x": np.asarray(d[f"{key}_x"], dtype=complex),
                       "y": np.asarray(d[f"{key}_y"], dtype=complex)}
    if "cal_rx_ref" in d:
        ex["cal_rx"] = {"ref": np.asarray(d["cal_rx_ref"], dtype=complex),
                        "obs": np.asarray(d["cal_rx_obs"], dtype=complex)}
    if "atten_ref" in d:
        ex["atten"] = {"ref": np.asarray(d["atten_ref"], dtype=complex),
                       "hi": np.asarray(d["atten_hi"], dtype=complex),
                       "lo": np.asarray(d["atten_lo"], dtype=complex),
                       "step_db": float(d["atten_step_db"])}
    if "op_conditions" in d:
        conditions = np.asarray(d["op_conditions"], dtype=float)
        pairs = []
        for i in range(len(conditions)):
            pairs.append((np.asarray(d[f"op{i}_x"], dtype=complex),
                          np.asarray(d[f"op{i}_y"], dtype=complex)))
        ex["operating_points"] = {"conditions": conditions,
                                  "pairs": pairs}
    return out


def extras_summary(extras: dict | None) -> list[dict]:
    """Checklist rows for a source's capture groups (GUI-friendly).

    One row per known group: ``{"group", "present", "n", "unlocks"}``
    where ``n`` is the sample count (or point count for
    operating_points).
    """
    extras = extras or {}
    rows = []
    for g in GROUPS:
        e = extras.get(g)
        if g == "operating_points":
            n = len(e["pairs"]) if e else 0
        elif e:
            n = len(e.get("x", e.get("ref", e.get("hi", []))))
        else:
            n = 0
        rows.append({"group": g, "present": e is not None, "n": n,
                     "unlocks": GROUP_UNLOCKS[g]})
    return rows
