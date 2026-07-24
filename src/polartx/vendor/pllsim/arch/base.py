# Vendored from pll_simulator@d7be4712: src/pllsim/arch/base.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Common architecture contract."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.jitter import ldbc_from_sphi
from ..core.results import AnalysisResult, SimResult


class PLLBase(ABC):
    """analyze() = linear phase-domain model; simulate() = behavioral time domain."""

    @abstractmethod
    def analyze(self, f: np.ndarray | None = None) -> AnalysisResult: ...

    @abstractmethod
    def simulate(self, n_cycles: int, *, noise: bool = True,
                 calibration: bool = True, seed: int = 0) -> SimResult: ...

    def design_report(self, ar: AnalysisResult | None = None) -> str:
        ar = ar or self.analyze()
        lines = [
            f"=== {type(self).__name__} design report ===",
            f"fout = {ar.f0 / 1e9:.6f} GHz",
            f"UGB = {ar.loop.f_ugb / 1e6:.3f} MHz   PM = {ar.loop.pm_deg:.1f} deg   "
            f"GM = {ar.loop.gm_db:.1f} dB",
            f"closed-loop f-3dB = {ar.loop.f_3db / 1e6:.3f} MHz   "
            f"peaking = {ar.loop.peaking_db:.2f} dB",
            f"RMS jitter ({ar.int_band[0]:.0f} Hz - {ar.int_band[1] / 1e6:.0f} MHz) "
            f"= {ar.jitter_fs:.1f} fs   IPN = {ar.ipn_dbc:.1f} dBc",
            f"dominant noise source: {ar.dominant_source()}",
        ]
        i = np.searchsorted(ar.f, 1e6)
        if i < ar.f.size:
            lines.append(f"L(1 MHz) total = {ldbc_from_sphi(ar.pn_breakdown['total'][i]):.1f} dBc/Hz")
        for k, v in ar.spurs_analytic.items():
            lines.append(f"spur[{k}] = {v:.1f} dBc (analytic)")
        for note in ar.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)
