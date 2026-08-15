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
    """The one block that selects the transmitter architecture.

    Everything else in the chain — CFR, the polar split, envelope
    normalization/skew/quantization, the DPA and every metric — is shared
    between the narrowband and wideband transmitters.  Implementing this
    interface is therefore the whole cost of adding an architecture, and
    it is also why narrowband and wideband results are directly
    comparable rather than "measured differently".

    Implementations in this package:

    ==========================  ====================================
    ``ADPLLTwoPoint``           narrowband: two-point modulated ADPLL
    ``DTCPhaseModulator``       wideband: open-loop DTC / phase interp
    ``IdealPhaseModulator``     passthrough, for chain bring-up
    ==========================  ====================================

    Contract for a new implementation:

    * ``modulate`` returns phase aligned sample-for-sample with the
      command — absorb any integer lag or linear trend internally.
    * Deterministic imperfections (quantization, INL, gain error) apply
      always; ``noise=False`` gates only the random terms.  This mirrors
      the pllsim engines and lets a caller separate the two.
    * If the model has a validity domain, state it in the docstring AND
      detect the violation at runtime (warn, and expose the offending
      quantity in ``diagnostics``).  A silent wrong answer is the failure
      mode this rule exists to prevent — see ``ADPLLTwoPoint`` event
      mode's per-reference-cycle phase advance check.
    """

    @abstractmethod
    def modulate(self, phase_cmd: np.ndarray, fs_bb: float, *,
                 noise: bool = True, seed: int = 0) -> PhaseModResult:
        """Transmit the commanded phase trajectory [rad] sampled at fs_bb."""


class IdealPhaseModulator(PhaseModulator):
    """Passthrough — chain bring-up and A/B reference."""

    def modulate(self, phase_cmd, fs_bb, *, noise=True, seed=0):
        return PhaseModResult(phase_out=np.asarray(phase_cmd, dtype=float),
                              diagnostics={"mode": "ideal"})
