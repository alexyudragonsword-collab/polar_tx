# Extracted from pll_simulator@5564c64: src/pllsim/arch/cppll.py (FracConfig only)
# so the vendored ADPLL does not drag in the analog charge-pump PLL blocks.
# Sole intentional divergence from upstream.
"""Fractional-N configuration shared by the DTC-based architectures."""
from __future__ import annotations

from dataclasses import dataclass

from ..blocks.dtc import DTCConfig
from ..core.deltasigma import Efm1, Mash11, Mash111


@dataclass
class FracConfig:
    """Fractional-N configuration."""

    frac: float                       # fractional part of N, [0, 1)
    mash_order: int = 3               # 1, 2 or 3
    bits: int = 24
    dtc: DTCConfig | None = None      # wired by blocks.dtc
    dtc_cal: object | None = None   # gain calibrator (LMSGainCal/SignSignLMS)
    dtc_lut_cal: object | None = None   # INL calibrator (LUTCal, seconds)

    def __post_init__(self):
        if not 0.0 <= self.frac < 1.0:
            raise ValueError(f"FracConfig frac must be in [0, 1), got {self.frac}")
        if self.mash_order not in (1, 2, 3):
            raise ValueError(f"FracConfig mash_order must be 1, 2 or 3, got {self.mash_order}")
        if not 1 <= int(self.bits) <= 32:
            raise ValueError(f"FracConfig bits must be in 1..32, got {self.bits}")

    def make_mash(self):
        return {1: Efm1, 2: Mash11, 3: Mash111}[self.mash_order](self.bits)

    @property
    def frac_word(self) -> int:
        return int(round(self.frac * (1 << self.bits)))
