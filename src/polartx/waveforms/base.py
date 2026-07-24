"""Common waveform container for the polar TX chain."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
    ofdm_ref: object | None = None          # padpd OFDMWaveform for kind="ofdm"
    freq_ideal: np.ndarray | None = None    # [Hz] on the fs grid, kind="gfsk"
    phase_ideal: np.ndarray | None = None   # [rad], kind="gfsk"
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.x.size
