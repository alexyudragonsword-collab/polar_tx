"""The front-end contract: what each preset lets a user turn, what each
reported metric means, and the plot data every renderer draws from.

``guiutil`` computes; this module describes.  The front ends render these
descriptions instead of each hand-writing a form, which is how three hand
written forms drifted apart and two of them sent a knob to a chain that
misread it (the DTC resolution reaching the RF-DAC as its DAC resolution).

Three rules hold the descriptions to the library:

* **Defaults are read, never typed.**  A knob's default is the default of
  the preset function's own keyword (``inspect.signature``), converted to
  display units; ``tests/test_guispec.py`` builds every preset from the
  form's defaults and requires the output to be bit-identical to the
  preset built with no overrides at all.
* **A knob must act.**  The same tests turn every knob once and require the
  output samples to change (a ``noise_only`` knob with noise on: jitter is
  a noise source, and with the two points matched the loop bandwidth only
  shapes noise), and an "ignored setting" warning fails.
* **Nothing here computes physics.**  Labels, units, ranges, unit
  conversion; the matplotlib renderer draws plot data it is handed.
"""
from __future__ import annotations

import inspect
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

#: knob groups, in display order: id -> (zh, en)
GROUPS: dict[str, tuple[str, str]] = {
    "phase": ("相位通路", "phase path"),
    "envelope": ("包络通路", "envelope path"),
    "output": ("输出级", "output stage"),
    "engine": ("引擎", "engine"),
}


@dataclass(frozen=True)
class Knob:
    """One control.  ``value`` is in display units; the preset receives
    ``offset + scale * value`` as keyword ``kw`` (or ``None`` for "off"
    on an ``allow_off`` knob, or the value itself for bool / choice)."""

    id: str
    kw: str
    kind: str                      # "int" | "float" | "bool" | "choice"
    zh: str
    en: str
    unit: str = ""
    group: str = "phase"
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    scale: float = 1.0
    offset: float = 0.0
    choices: tuple[tuple[str, str, str], ...] = ()   # (value, zh, en)
    allow_off: bool = False
    #: acts only when the run's noise is on: a noise source itself, or a
    #: loop that, with the two points matched, only shapes noise
    noise_only: bool = False
    help_zh: str = ""
    help_en: str = ""
    default: Any = None            # display units; filled per preset

    def to_override(self, v: Any) -> Any:
        if self.allow_off and v == "off":
            return None
        if self.kind == "bool":
            return bool(v)
        if self.kind == "choice":
            if v not in [c[0] for c in self.choices]:
                raise ValueError(f"{self.id}: {v!r} is not one of "
                                 f"{[c[0] for c in self.choices]}")
            return v
        if self.kind == "int":
            return int(round(self.offset + self.scale * float(v)))
        return self.offset + self.scale * float(v)

    def from_override(self, ov: Any) -> Any:
        if ov is None and self.allow_off:
            return "off"
        if self.kind in ("bool", "choice"):
            return ov
        v = (float(ov) - self.offset) / self.scale
        return int(round(v)) if self.kind == "int" else round(v, 9)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["choices"] = [list(c) for c in self.choices]
        return d


# --------------------------------------------------------------- the knobs
_SKEW_HELP = ("包络通路相对相位通路的时延。OFDM 在 1 ns（LTE-20）/ 0.2 ns（WiFi 160）"
              "就掉 mask；准恒包络 EDR 能容忍几十 ns",
              "envelope-path delay against the phase path. OFDM fails its "
              "mask at 1 ns (LTE-20) / 0.2 ns (WiFi 160 MHz); quasi-constant-"
              "envelope EDR tolerates tens of ns")
_KNOBS: dict[str, Knob] = {k.id: k for k in (
    Knob("dtc_bits", "n_bits", "int", "DTC 位数", "DTC bits", "bit",
         "phase", 6, 14, 1),
    Knob("dither", "dither", "bool", "DTC dither", "DTC dither", "", "phase",
         help_zh="把 DTC 量化误差从杂散打成噪底",
         help_en="turns DTC quantization spurs into a noise floor"),
    Knob("jitter_fs", "jitter_rms_s", "float", "DTC 抖动", "DTC jitter",
         "fs rms", "phase", 0, 500, 10, scale=1e-15, noise_only=True),
    Knob("dp_err_pct", "dp_gain", "float", "两点增益误差",
         "two-point gain error", "%", "phase", -10, 10, 0.5,
         scale=0.01, offset=1.0,
         help_zh="直接调制点相对环路的增益失配",
         help_en="direct-modulation path gain against the loop"),
    Knob("loop_bw_khz", "loop_bw", "float", "环路带宽", "loop bandwidth",
         "kHz", "phase", 50, 400, 10, scale=1e3, noise_only=True,
         help_zh="两点匹配时只整形噪声；设一个增益误差才能看到它与失配的取舍",
         help_en="with the two points matched this shapes only the noise; "
                 "set a gain error to see it trade against mismatch"),
    Knob("skew_ns", "env_skew_s", "float", "AM/PM skew", "AM/PM skew",
         "ns", "envelope", -4, 4, 0.05, scale=1e-9,
         help_zh=_SKEW_HELP[0], help_en=_SKEW_HELP[1]),
    Knob("env_floor", "env_floor", "float", "包络下限（打孔）",
         "envelope floor (hole punch)", "", "envelope", 0.0, 0.2, 0.01),
    Knob("cfr_db", "cfr_papr_db", "float", "CFR 目标 PAPR", "CFR target PAPR",
         "dB", "envelope", 5, 12, 0.5, allow_off=True),
    Knob("dpd", "dpd", "bool", "极坐标 DPD", "polar DPD", "", "envelope"),
    Knob("dac_bits", "n_bits", "int", "DAC 位数", "DAC bits", "bit",
         "output", 6, 16, 1,
         help_zh="每轴幅度分辨率（RF-DAC）",
         help_en="per-axis magnitude resolution (RF-DAC)"),
    Knob("mode", "mode", "choice", "ADPLL 引擎", "ADPLL engine", "", "engine",
         choices=(("response", "response（线性化，快；不背书杂散）",
                   "response (linearized, fast; not for spur conclusions)"),
                  ("event", "event（逐参考周期）",
                   "event (per reference cycle)"))),
)}


_Entry = tuple[tuple[Callable, ...], tuple[str, ...], dict[str, dict]]


def _presets_table() -> dict[str, _Entry]:
    """name -> (functions whose signatures hold the defaults, in lookup
    order; knob ids; per-preset field changes).  Lazy so importing this
    module does not import the library."""
    from . import presets as P
    dtc_polar = ("dtc_bits", "dither", "jitter_fs", "skew_ns", "env_floor",
                 "cfr_db", "dpd")
    adpll = ("dp_err_pct", "loop_bw_khz", "mode")
    edr_skew = {"skew_ns": {"lo": -20, "hi": 20, "step": 0.5}}
    lte_bw = {"loop_bw_khz": {"lo": 200, "hi": 3000, "step": 100}}
    wifi: _Entry = ((P.wifi_dtc,), dtc_polar, {})
    nr: _Entry = ((P.nr_dtc, P.wifi_dtc), dtc_polar, {})
    return {
        "BLE LE-1M": ((P.ble_adpll,), adpll, {}),
        "BLE LE-2M": ((P.ble_adpll,), adpll, {}),
        "BT EDR2 pi/4-DQPSK": ((P.bt_edr_adpll,), adpll + ("skew_ns",),
                               edr_skew),
        "BT EDR3 8DPSK": ((P.bt_edr_adpll,), adpll + ("skew_ns",), edr_skew),
        "LTE 20 MHz": ((P.lte20_adpll,),
                       adpll + ("skew_ns", "env_floor", "cfr_db", "dpd"),
                       lte_bw),
        "WiFi 80 MHz": wifi, "WiFi 160 MHz": wifi, "WiFi 320 MHz": wifi,
        "NR FR1 100 MHz": nr, "NR FR2 200 MHz": nr,
        # no envelope path: the skew, the hole punch and the polar DPD are
        # neutralized in the branches (outphasing.py), so not offered
        "WiFi 160 MHz (outphasing)": ((P.wifi_outphasing, P.wifi_dtc),
                                      ("dtc_bits", "dither", "jitter_fs",
                                       "cfr_db"), {}),
        # its n_bits is the DAC's (12), a different knob from the DTC's
        "WiFi 160 MHz (RF-DAC)": ((P.wifi_rfdac, P.wifi_dtc),
                                  ("dac_bits", "cfr_db"), {}),
    }


def _signature_default(fns: tuple[Callable, ...], kw: str) -> Any:
    for fn in fns:
        p = inspect.signature(fn).parameters.get(kw)
        if p is not None and p.default is not inspect.Parameter.empty:
            return p.default
    raise LookupError(f"no default for {kw!r} in "
                      f"{[f.__name__ for f in fns]}")


def knob_specs(name: str) -> list[Knob]:
    """The controls preset ``name`` offers, with its own defaults in
    display units.  Benchmarks offer none (they fix their published-class
    parameters); an unknown name raises."""
    table = _presets_table()
    if name not in table:
        from .guiutil import PRESETS
        if name in PRESETS:
            return []
        raise ValueError(f"unknown preset {name!r}")
    fns, ids, changes = table[name]
    out = []
    for kid in ids:
        k = replace(_KNOBS[kid], **changes.get(kid, {}))
        out.append(replace(k, default=k.from_override(
            _signature_default(fns, k.kw))))
    return out


def overrides_from_ui(name: str, values: dict[str, Any]) -> dict[str, Any]:
    """Display-unit control values (by knob id) -> preset keywords.

    Ids the preset does not offer are dropped, so a front end may send its
    whole form.  ``"off"`` turns an ``allow_off`` knob off."""
    out: dict[str, Any] = {}
    for k in knob_specs(name):
        if k.id in values and values[k.id] is not None:
            out[k.kw] = k.to_override(values[k.id])
    return out


# ------------------------------------------------------------- the metrics
#: report metric key -> (zh, en, unit, better: "lower"|"higher"|None,
#: kind: "number"|"verdict"|"text")
_METRICS: dict[str, tuple[str, str, str, str | None, str]] = {
    "EVM [dB]": ("EVM", "EVM", "dB", "lower", "number"),
    "EVM [%]": ("EVM", "EVM", "%", "lower", "number"),
    "DEVM [%]": ("DEVM", "DEVM", "%", "lower", "number"),
    "phase EVM [%]": ("相位 EVM", "phase EVM", "%", "lower", "number"),
    "ACLR upper [dBc]": ("ACLR 上邻道", "ACLR upper", "dBc", "lower",
                         "number"),
    "freq deviation avg [kHz]": ("频偏均值（本 burst）",
                                 "freq deviation avg (this burst)", "kHz",
                                 None, "number"),
    "drift [Hz/us]": ("载波漂移率", "carrier drift rate", "Hz/us", None,
                      "number"),
    "mask": ("频谱 mask", "spectral mask", "", None, "verdict"),
    "mask OOB margin [dB]": ("mask 带外余量", "mask out-of-band margin", "dB",
                             "higher", "number"),
    "CPE rms [deg]": ("公共相位误差", "common phase error", "deg rms",
                      "lower", "number"),
    "EVM if CPE tracked [dB]": ("跟踪 CPE 后的 EVM", "EVM if CPE tracked",
                                "dB", "lower", "number"),
    "EVM per-tone eq [dB]": ("按子载波均衡后的 EVM", "EVM, per-tone equalized",
                             "dB", "lower", "number"),
    "constellation": ("星座判读", "constellation", "", None, "text"),
}
_INBAND = re.compile(r"^in-band \+-([0-9.]+) MHz \[dBc\]$")


def metric_info(key: str) -> dict:
    """Label, unit and direction of a metric key ``run_chain_report``
    produces.  Raises on a key nobody described, so a new metric cannot
    reach the cards unlabeled."""
    row: tuple[str, str, str, str | None, str]
    m = _INBAND.match(key)
    if m:
        off = m.group(1)
        row = (f"带内发射 ±{off} MHz", f"in-band emission ±{off} MHz",
               "dBc", "lower", "number")
    elif key in _METRICS:
        row = _METRICS[key]
    else:
        raise KeyError(f"metric {key!r} has no description in guispec")
    zh, en, unit, better, kind = row
    return {"zh": zh, "en": en, "unit": unit, "better": better, "kind": kind}


# --------------------------------------------------------------- the plots
def decimate_peak(x, y, n_max: int = 2048):
    """Downsample a spectrum for display by keeping each bucket's maximum
    — a spectrum analyser's max-hold, so a spur or a mask violation never
    disappears between kept points.  Verdicts are computed on the full
    arrays before this; only the drawing is thinned."""
    import numpy as np
    x, y = np.asarray(x), np.asarray(y)
    k = int(np.ceil(x.size / n_max))
    if k <= 1:
        return x, y
    n = (x.size // k) * k
    xs = x[:n].reshape(-1, k)
    ys = y[:n].reshape(-1, k)
    idx = ys.argmax(axis=1)
    rows = np.arange(xs.shape[0])
    return xs[rows, idx], ys[rows, idx]


def thin_points(z, n_max: int = 4000):
    """Every k-th point of a scatter, so a 1024-QAM burst crosses the
    phone bridge at a few thousand points instead of tens of thousands."""
    import numpy as np
    z = np.asarray(z).ravel()
    k = int(np.ceil(z.size / n_max))
    return z[::k] if k > 1 else z


def compact_plots(plots: list[dict]) -> list[dict]:
    """Plot data rounded to five significant digits per series, for the
    bridge: a few hundred kilobytes of 17-digit floats per run is the
    difference between a slider that follows the finger and one that
    does not.  The matplotlib figure is drawn from the unrounded data."""
    import numpy as np

    def rnd(a):
        a = np.asarray(a, dtype=float)
        m = float(np.nanmax(np.abs(a))) if a.size else 0.0
        dec = max(0, 4 - int(np.floor(np.log10(m)))) if m > 0 else 0
        return np.round(a, dec)

    out = []
    for pl in plots:
        q = {k: v for k, v in pl.items() if k != "series"}
        q["series"] = [{**{k: v for k, v in s.items() if k != "mpl"},
                        "x": rnd(s["x"]), "y": rnd(s["y"])}
                       for s in pl["series"]]
        out.append(q)
    return out


_STYLE: dict[str, dict[str, Any]] = {"trace": dict(lw=0.7), "mask": dict(color="r", ls="--", lw=1.0),
          "points": dict(ls="none", marker=".", ms=1, alpha=0.4),
          "ref": dict(color="k", ls=":", lw=1.0),
          "line": dict(lw=0.8)}


def render_matplotlib(plots: list[dict], figsize=(11, 4.2)):
    """Draw report plot data as one matplotlib figure, side by side —
    the PNG for the desktop GUIs, the examples and the phone fallback.
    Draws only what it is given."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(plots), figsize=figsize, squeeze=False)
    for ax, pl in zip(axs[0], plots):
        for s in pl["series"]:
            st: dict[str, Any] = dict(_STYLE[s.get("style", "line")])
            if "color" in s:                   # several traces on one panel
                st["color"] = f"C{int(s['color'])}"
            st.update(s.get("mpl", {}))
            ax.plot(s["x"], s["y"], label=s.get("label"), **st)
        if pl.get("legend"):
            ax.legend(fontsize=8)
        ax.set(xlabel=pl.get("xlabel", ""), ylabel=pl.get("ylabel", ""))
        ax.set_title(pl["title"], fontsize=pl.get("title_size", None))
        if pl.get("xlim"):
            ax.set_xlim(*pl["xlim"])
        if pl.get("ylim"):
            ax.set_ylim(*pl["ylim"])
        if pl.get("grid"):
            ax.grid(True, alpha=0.3)
        if pl.get("equal"):
            ax.set_aspect("equal")
    fig.tight_layout()
    return fig
