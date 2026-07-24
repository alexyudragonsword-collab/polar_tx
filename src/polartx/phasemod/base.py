"""Phase-modulator interface: the block the two TX flavors swap."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PhaseModResult:
    """Actual transmitted phase on the baseband grid.

    phase_out is aligned sample-for-sample with the commanded phase (any
    integer-lag/linear-trend bookkeeping is done inside the modulator);
    it includes the modulator's quantization, INL, mismatch residue and
    (noise=True) phase noise.  diagnostics carries engine traces.
    """

    phase_out: np.ndarray
    diagnostics: dict = field(default_factory=dict)


class PhaseModulator(ABC):
    @abstractmethod
    def modulate(self, phase_cmd: np.ndarray, fs_bb: float, *,
                 noise: bool = True, seed: int = 0) -> PhaseModResult:
        """Transmit the commanded phase trajectory [rad] sampled at fs_bb."""


class IdealPhaseModulator(PhaseModulator):
    """Passthrough — chain bring-up and A/B reference."""

    def modulate(self, phase_cmd, fs_bb, *, noise=True, seed=0):
        return PhaseModResult(phase_out=np.asarray(phase_cmd, dtype=float),
                              diagnostics={"mode": "ideal"})
