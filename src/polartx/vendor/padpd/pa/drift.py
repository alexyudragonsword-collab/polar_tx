# Vendored from PA_DPD@44cbcb3: src/padpd/pa/drift.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Time-varying PA for DPD tracking studies.

A :class:`~padpd.pa.reference_pa.ReferencePA` whose operating point and
nonlinearity drift with an external "state" scalar (0..1) standing in for
temperature rise / supply sag / aging. Advancing the state changes:

- ``drive``      : bias/gain drift moves the compression point;
- ``beta_a``     : the Saleh AM-AM knee sharpens (more compression hot);
- ``alpha_p``    : AM-PM slope grows.

This lets an adaptive DPD be tested against a PA that changes underneath
it, the way a real field unit does.
"""

from __future__ import annotations

import numpy as np

from .reference_pa import ReferencePA
from .saleh import SalehPA


class DriftingReferencePA:
    def __init__(self, drive0: float = 0.13, drive_span: float = 0.03,
                 beta_a_span: float = 0.25, alpha_p_span: float = 0.8):
        self.drive0 = drive0
        self.drive_span = drive_span
        self.beta_a_span = beta_a_span
        self.alpha_p_span = alpha_p_span
        self._base = SalehPA()
        self.state = 0.0

    def set_state(self, state: float) -> None:
        """Set the drift state in [0, 1] (0 = cold/new, 1 = hot/aged)."""
        self.state = float(np.clip(state, 0.0, 1.0))

    def pa(self) -> ReferencePA:
        """The frozen PA at the current drift state."""
        s = self.state
        saleh = SalehPA(alpha_a=self._base.alpha_a,
                        beta_a=self._base.beta_a * (1 + self.beta_a_span * s),
                        alpha_p=self._base.alpha_p
                        * (1 + self.alpha_p_span * s),
                        beta_p=self._base.beta_p)
        return ReferencePA(drive=self.drive0 + self.drive_span * s,
                           saleh=saleh)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.pa()(x)
