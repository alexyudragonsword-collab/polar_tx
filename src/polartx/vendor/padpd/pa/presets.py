# Extracted from PA_DPD@44cbcb3: src/padpd/pa/presets.py (gmp_opendpd_510 / mp_opendpd_500 only; ddr_volterra_default dropped — DDRVolterraModel is not vendored)
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Model configurations matching OpenDPD's published classical baselines.

OpenDPD's benchmark (benchmark/benchmark_volterra_qr.py and
benchmark/benchmark_report.md there) compares neural models against
least-squares MP/GMP at a ~500-real-parameter budget. These factories
reproduce those exact configurations so padpd results are directly
comparable with the published numbers. (One complex coefficient counts
as 2 real parameters.)
"""

from __future__ import annotations

from .gmp import GMPModel
from .memory_polynomial import MemoryPolynomialModel


def mp_opendpd_500() -> MemoryPolynomialModel:
    """OpenDPD benchmark MP: K=5, Q=50 -> 250 complex = 500 real params."""
    return MemoryPolynomialModel(order=5, memory_depth=50)


def gmp_opendpd_510() -> GMPModel:
    """OpenDPD benchmark GMP (Morgan 2006): Ka=5/La=15, Kb=4/Lb=15/Mb=2,
    Kc=4/Lc=15/Mc=1 -> 255 complex = 510 real params."""
    return GMPModel(order=5, memory_depth=15,
                    lag_order=4, lag_memory=15, lag_span=2,
                    lead_order=4, lead_memory=15, lead_span=1)
