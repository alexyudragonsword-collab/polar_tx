# Vendored from PA_DPD@44cbcb3: src/padpd/pa/reference_pa.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Virtual "device under test" PA for synthetic data generation.

A Wiener-Hammerstein structure (FIR -> Saleh nonlinearity -> FIR) with
fixed coefficients. It plays the role of the transistor-level PA
(Cadence Spectre envelope simulation) until real EDA data is plugged in:
downstream modeling/DPD code treats it as a black box x -> y.

The structure is intentionally NOT in the GMP model class, so fitted GMP
models approximate it (realistic residual error) rather than match exactly.

Input is assumed to be normalized to unit average power; ``drive`` sets
the operating point (larger = closer to saturation). Small-signal gain is
normalized to ~1 so absolute gain drops out of modeling and metrics.

Note on invertibility: DPD can only linearize up to the PA's maximum
output. With unit-power OFDM (PAPR ~10-11 dB) the peak envelope is
~3.3-3.8, and the Saleh AM-AM turns over at r = 1/sqrt(beta_a). Keep
``drive`` <= ~0.15 so that peaks stay in the invertible region; around
0.22 and above the ILA DPD diverges at the peaks (a realistic failure
mode — real systems add crest factor reduction or more back-off).

This class is a *virtual test article*, deliberately NOT persistable via
``PAModel.save`` (no ``get_config``): it stands in for a transistor-level
simulation, not for a fitted model worth serializing. The same applies
to :class:`~padpd.pa.drift.DriftingReferencePA`.
"""

from __future__ import annotations

import numpy as np

from .base import PAModel
from .saleh import SalehPA


class ReferencePA(PAModel):
    def __init__(self, drive: float = 0.14,
                 fir_in: np.ndarray | None = None,
                 fir_out: np.ndarray | None = None,
                 saleh: SalehPA | None = None):
        self.drive = drive
        self.fir_in = (np.array([1.0, 0.12 - 0.05j, 0.04 + 0.03j])
                       if fir_in is None else np.asarray(fir_in))
        self.fir_out = (np.array([1.0, 0.06 + 0.025j])
                        if fir_out is None else np.asarray(fir_out))
        self.saleh = saleh or SalehPA()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        u = np.convolve(x, self.fir_in)[: len(x)]
        v = self.saleh(self.drive * u)
        w = np.convolve(v, self.fir_out)[: len(x)]
        return w / (self.drive * self.saleh.small_signal_gain)
