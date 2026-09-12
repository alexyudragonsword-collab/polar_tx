"""Common waveform container for the polar TX chain."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Imported for typing only: the annotation is what the comment on
    # ofdm_ref used to say in prose, and a type checker can read this one.
    # Keeping it out of runtime also keeps the vendor tree off this
    # module's import path.
    from ..vendor.padpd.waveform.ofdm import OFDMWaveform


@dataclass
class Waveform:
    """Complex-baseband stimulus plus the references the metrics need.

    x is unit-average-power; fs the sample rate; bw the channel bandwidth
    (metric conventions key off it).  kind selects the metric set:
    "ofdm" (EVM via demod against ofdm_ref) or "gfsk" (phase-trajectory
    EVM / frequency-deviation metrics via freq_ideal/phase_ideal).
    """

    x: np.ndarray
    fs: float
    bw: float
    kind: str
    ofdm_ref: OFDMWaveform | None = None    # demod reference for kind="ofdm"
    freq_ideal: np.ndarray | None = None    # [Hz] on the fs grid, kind="gfsk"
    phase_ideal: np.ndarray | None = None   # [rad], kind="gfsk"
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Number of baseband samples in the burst."""
        return self.x.size

    def require_ofdm_ref(self) -> OFDMWaveform:
        """The demodulation reference, or a clear error.

        A metric that needs the reference raises when it is missing instead
        of limping on (CONTRIBUTING.md 6): the absence means the caller built
        the waveform wrong, and the alternative is an exception thrown deep
        inside a demodulator — or worse, a NaN that travels into a report
        looking like a measurement.
        """
        if self.ofdm_ref is None:
            raise ValueError(f'waveform kind="{self.kind}" carries no '
                             "ofdm_ref: nothing to demodulate against")
        return self.ofdm_ref
