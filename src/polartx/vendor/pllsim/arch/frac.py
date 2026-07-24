# Extracted from pll_simulator@d7be4712: src/pllsim/arch/cppll.py (FracConfig,
# frac_spur_offsets only) so the vendored ADPLL does not drag in the analog
# charge-pump PLL blocks.  Sole intentional divergence from upstream.
"""Fractional-N configuration shared by the DTC-based architectures."""
from __future__ import annotations

from dataclasses import dataclass

from ..core.deltasigma import Efm1, Mash11, Mash111


def frac_spur_offsets(frac: float, fref: float, kmax: int = 6,
                      fmin: float = 1e3) -> list[float]:
    """Expected fractional-spur offsets: k*frac folded into [0, fref/2]."""
    offs = set()
    for k in range(1, kmax + 1):
        x = (k * frac) % 1.0
        fo = min(x, 1.0 - x) * fref
        if fmin < fo < 0.45 * fref:
            offs.add(round(fo, 3))
    return sorted(offs)


@dataclass
class FracConfig:
    """Fractional-N configuration."""

    frac: float                       # fractional part of N, [0, 1)
    mash_order: int = 3               # 1, 2 or 3
    bits: int = 24
    dtc: "object | None" = None       # DTCConfig, wired by blocks.dtc
    dtc_cal: "object | None" = None   # gain calibrator (LMSGainCal/SignSignLMS)
    dtc_lut_cal: "object | None" = None   # INL calibrator (LUTCal, seconds)

    def make_mash(self):
        return {1: Efm1, 2: Mash11, 3: Mash111}[self.mash_order](self.bits)

    @property
    def frac_word(self) -> int:
        return int(round(self.frac * (1 << self.bits)))
