"""Is a wideband run's output converged on its simulation grid?

The DTC and DPA update at a clock; what leaves the chip is their output
held between updates.  A grid equal to the clock reads the output at the
update instants only: the hold's sinc droop is missing and its images,
which fall at multiples of the clock, land exactly on the signal and are
invisible.  ``ChainConfig.f_clock`` separates the two rates; this module
says, per run, whether the grid it was read on is fine enough -- the same
call array_tx makes for its engines (``report.disclaimer_for``), in the
same words, with the deviations polar_tx measured itself
(examples/ex22_grid_convergence.py, cairn/wideband-grid.md).
"""
from __future__ import annotations

from typing import Any

#: grid samples per update at which ex22 found ACLR1 / ACLR2, the mask
#: margin over +-2 BW and both EVMs converged (os 16 on the presets' 4x
#: clock: os 32 within 0.2 dB of it).  The image level the full-span mask
#: sees converges slower, as the discrete hold approaches the continuous
#: one: os 32 is 0.55 dB from os 16.
CONVERGED_HOLD = 4

#: every disclaimer carries this phrase, as array_tx's do
NOT_EVIDENCE = "不能背书"

DISCLAIMERS = {
    "held": {
        "zh": ("采样网格未收敛（每次更新 < 4 个网格点，即预设的 4× 时钟上 os < 16："
               "ACLR1 偏保守 0.3–1.0 dB、ACLR2 偏保守 1.8–2.4 dB、模板裕量偏保守 0.5–2.9 dB "
               "且只看到 ±fs/2，更新时钟镜像（首镜像约 −17 dBc）不在网格上；"
               "scalar EVM 少算保持下垂（偏乐观至多 2.6 dB），per-tone EVM 已收敛，ex22），"
               "不能背书：杂散 / 模板结论只做趋势比较"),
        "en": ("grid not converged (< 4 grid samples per update, os < 16 on the "
               "presets' 4x clock: ACLR1 0.3-1.0 dB pessimistic, ACLR2 1.8-2.4 dB, "
               "mask margin 0.5-2.9 dB and only +-fs/2 seen, the update-clock "
               "images (first ~-17 dBc) off the grid; scalar EVM misses the hold's "
               "droop, up to 2.6 dB optimistic; per-tone EVM converged, ex22): "
               "spur / mask conclusions are trend-only, not evidence"),
    },
    "tied": {
        "zh": ("更新时钟随网格（未与网格分开）：提高 os 换的是更快的硬件，不是更细的网格；"
               "在这个时钟上读，保持下垂与镜像不在网格上，网格收敛未验证，"
               "不能背书：杂散 / 模板结论只做趋势比较"),
        "en": ("update clock tied to the grid: raising os clocks the hardware "
               "faster rather than refining the grid; read at that clock, the "
               "hold's droop and images are off the grid and grid convergence is "
               "unverified: spur / mask conclusions are trend-only, not evidence"),
    },
}

CONVERGED_NOTE = {
    "zh": "采样网格已收敛（每次更新 ≥ 4 个网格点，ex22：与 8 个点差 ≤ 0.2 dB；全幅模板的镜像电平差 0.55 dB）",
    "en": ("grid converged (>= 4 grid samples per update, ex22: within 0.2 dB "
           "of 8; the image level on the full-span mask within 0.55 dB)"),
}


def _chain(tx: Any) -> tuple[Any, Any, bool]:
    """(chain config, phase modulator, is the polar chain itself) of a
    polar chain or one of the topologies that wrap one."""
    inner = getattr(tx, "tx", None)            # outphasing / FIR wrap a PolarTX
    if inner is not None and hasattr(inner, "cfg"):
        return inner.cfg, getattr(inner, "phasemod", None), False
    return getattr(tx, "cfg", None), getattr(tx, "phasemod", None), True


def grid_disclaimer(tx: Any, fs: float) -> dict | None:
    """The grid note one wideband run needs: ``{"id": "grid", "zh", "en",
    "converged"}``, or None for a chain this does not apply to (the ADPLL,
    whose output is the loop's, not a held code).

    ``fs`` is the grid the output was read on (``PolarResult.fs``)."""
    from .chain import clock_hold
    from .phasemod.adpll_tp import ADPLLTwoPoint
    cfg, pm, polar = _chain(tx)
    if cfg is None or isinstance(pm, ADPLLTwoPoint):
        return None
    f_clock = getattr(cfg, "f_clock", None)
    hold = 1 if f_clock is None else clock_hold(fs, f_clock)
    if hold >= CONVERGED_HOLD:
        return {"id": "grid", "converged": True, **CONVERGED_NOTE}
    key = "held" if (polar and f_clock is not None) else "tied"
    return {"id": "grid", "converged": False, **DISCLAIMERS[key]}


def grid_verdict(series: dict, tol: dict, *, base: int = 4, ref: int = 16,
                 check: int = 32) -> dict:
    """A convergence verdict per metric, array_tx's protocol: the reading
    at ``base`` converges when it is within ``tol[metric]`` of the one at
    ``ref``, and ``ref`` is itself converged when ``check`` is within
    ``tol`` of it.  ``series`` maps oversampling -> {metric: value}; a
    value of None (a band the grid cannot hold) gives None, never a pass.

    Returns {metric: {"d_base", "d_check", "tol", "base_ok", "ref_ok"}},
    the differences signed as reading minus reference."""
    out = {}
    for key, t in tol.items():
        r, c, b = series[ref][key], series[check][key], series[base][key]
        d_base = None if b is None or r is None else b - r
        d_check = None if c is None or r is None else c - r
        out[key] = {"d_base": d_base, "d_check": d_check, "tol": t,
                    "base_ok": None if d_base is None else abs(d_base) <= t,
                    "ref_ok": None if d_check is None else abs(d_check) <= t}
    return out
