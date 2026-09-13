# Vendored from pll_simulator@931cfaf: src/pllsim/core/results.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Structured results consumed by plotting and reports."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .freqresp import FreqResponse, LoopMetrics


@dataclass
class AnalysisResult:
    """Frequency-domain (linear phase-domain model) analysis output."""

    f: np.ndarray                                  # offset grid [Hz]
    f0: float                                      # output carrier [Hz]
    pn_breakdown: dict[str, np.ndarray]            # per-source S_phi [rad^2/Hz] + 'total'
    loop: LoopMetrics
    jitter_fs: float                               # RMS jitter over the integration band
    ipn_dbc: float
    int_band: tuple[float, float] = (1e3, 100e6)
    spurs_analytic: dict[str, float] = field(default_factory=dict)   # name -> dBc
    ntfs: dict[str, FreqResponse] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def ipn_shares(self) -> list[tuple[str, float, float]]:
        """Per-source (name, share of integrated phase power, jitter [fs]).

        Sorted worst first.  The shares sum to 1 because the sources are
        uncorrelated and `total` is their sum -- checked to 1e-6 on every
        benchmark preset, which is what lets this be drawn as a pie at all.

        The *share* is of power, not of jitter: a source holding half the
        power contributes 1/sqrt(2) of the total RMS jitter, not half of it,
        which is why the per-source jitter is returned alongside rather than
        left for the reader to divide.
        """
        from .jitter import integrate_pn
        rows = []
        for k, s in self.pn_breakdown.items():
            if k == "total":
                continue
            p = integrate_pn(self.f, s, *self.int_band)
            rows.append((k, p, 1e15 * math.sqrt(p) / (2.0 * math.pi * self.f0)))
        tot = sum(p for _k, p, _j in rows)
        if tot <= 0.0:
            return [(k, 0.0, j) for k, _p, j in rows]
        return sorted(((k, p / tot, j) for k, p, j in rows),
                      key=lambda r: -r[1])

    def dominant_source(self) -> str:
        """Source with the largest contribution to integrated phase power."""
        shares = self.ipn_shares()
        return shares[0][0] if shares else ""


@dataclass
class SimResult:
    """Time-domain simulation output (reference-edge granularity unless noted)."""

    fs: float                                      # sample rate of the sequences [Hz]
    f0: float                                      # output carrier [Hz]
    t: np.ndarray                                  # sample times [s]
    phase_err_out: np.ndarray                      # output-referred phase error [rad]
    freq_out: np.ndarray                           # instantaneous output freq [Hz]
    ctrl: np.ndarray                               # vctrl [V] or OTW [LSB]
    cal_traces: dict[str, np.ndarray] = field(default_factory=dict)
    lock_time_s: float | None = None
    f_psd: np.ndarray | None = None                # cached periodogram (settled portion)
    s_phi_psd: np.ndarray | None = None
    spurs_fft: dict[float, float] = field(default_factory=dict)
    jitter_fs: float | None = None                 # from time-domain PSD if computed
    notes: list[str] = field(default_factory=list)  # caveats about the numbers
    extra: dict = field(default_factory=dict)

    def reported_psd(self) -> tuple[str, np.ndarray, np.ndarray, float]:
        """The record ``jitter_fs`` came from: ``(name, f, S_phi, fs)``.

        Anything that shows a measured spectrum has to read this, so that the
        picture and the printed number can never describe different runs.
        They did, for a year: the MDLL drew its reference-rate record -- which
        edge replacement leaves nearly flat, because that record samples the
        phase exactly where it was just realigned -- against a model of the
        real output phase.  8-11 dB apart on all three front ends, while the
        cross-domain sweep asked for the oversampled record by hand, read
        2-3 dB, and stayed green.

        The answer is derived rather than stored because ``arch.base
        .attach_fine`` is the only writer of the fine record and it replaces
        ``jitter_fs`` in the same breath: the record's presence *is* the
        statement that it, not the reference-rate one, is what this result
        reports.  A separate flag would be a second thing to forget.
        """
        extra = self.extra or {}
        fine_f = extra.get("fine_f_avg", extra.get("fine_f"))
        if fine_f is not None:
            fine_s = extra.get("fine_psd_avg", extra.get("fine_psd"))
            return ("fine", np.asarray(fine_f), np.asarray(fine_s),
                    float(extra["fine_fs"]))
        if self.f_psd is None or self.s_phi_psd is None:
            raise ValueError(
                "this simulation carries no PSD (record too short?)")
        return "ref", self.f_psd, self.s_phi_psd, float(self.fs)
