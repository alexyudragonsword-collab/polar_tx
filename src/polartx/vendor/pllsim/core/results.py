# Vendored from pll_simulator@d7be4712: src/pllsim/core/results.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Structured results consumed by plotting and reports."""
from __future__ import annotations

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

    def dominant_source(self) -> str:
        """Source with the largest contribution to integrated phase power."""
        from .jitter import integrate_pn
        best, best_p = "", -1.0
        for k, s in self.pn_breakdown.items():
            if k == "total":
                continue
            p = integrate_pn(self.f, s, *self.int_band)
            if p > best_p:
                best, best_p = k, p
        return best


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
    extra: dict = field(default_factory=dict)
