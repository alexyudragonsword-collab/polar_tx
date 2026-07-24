# Vendored from PA_DPD@44f9ee99: src/padpd/data/dataset.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""IQ input/output dataset container.

The canonical in-memory format for PA modeling and DPD training: a pair of
aligned complex baseband sequences (PA input ``x``, PA output ``y``) plus
the sample rate and free-form metadata. Every external source (synthetic,
Cadence envelope export, MATLAB capture, OpenDPD dataset) is converted to
this container so downstream code has a single interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class IQDataset:
    x: np.ndarray  # PA input, complex baseband
    y: np.ndarray  # PA output, complex baseband, sample-aligned with x
    sample_rate_hz: float
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=complex)
        self.y = np.asarray(self.y, dtype=complex)
        if self.x.shape != self.y.shape or self.x.ndim != 1:
            raise ValueError("x and y must be 1-D arrays of equal length")

    def __len__(self) -> int:
        return len(self.x)

    def split(self, fractions=0.8) -> tuple["IQDataset", ...]:
        """Contiguous split (preserves memory-effect continuity).

        ``fractions`` may be a single float f (two-way split f / 1-f) or a
        sequence like (0.6, 0.2, 0.2) for a train/val/test split (OpenDPD
        convention); the last part receives all remaining samples.
        """
        if np.isscalar(fractions):
            fractions = (float(fractions), 1.0 - float(fractions))
        if sum(fractions) > 1.0 + 1e-9:
            raise ValueError("split fractions must sum to <= 1")
        names = (("train", "test") if len(fractions) == 2
                 else ("train", "val", "test") if len(fractions) == 3
                 else tuple(f"part{i}" for i in range(len(fractions))))
        parts = []
        start = 0
        for i, frac in enumerate(fractions):
            stop = len(self) if i == len(fractions) - 1 else \
                start + int(len(self) * frac)
            parts.append(IQDataset(self.x[start:stop], self.y[start:stop],
                                   self.sample_rate_hz,
                                   {**self.meta, "split": names[i]}))
            start = stop
        return tuple(parts)

    def normalized(self) -> "IQDataset":
        """Return a copy with x scaled to unit average power.

        y is scaled by the same factor so the PA gain is preserved.
        The scale factor is recorded in meta["norm_scale"].
        """
        s = float(np.sqrt(np.mean(np.abs(self.x) ** 2)))
        return IQDataset(self.x / s, self.y / s, self.sample_rate_hz,
                         {**self.meta, "norm_scale": s})

    def save(self, path: str) -> None:
        np.savez_compressed(path, x=self.x, y=self.y,
                            sample_rate_hz=self.sample_rate_hz,
                            meta=np.array(repr(self.meta)))

    @classmethod
    def load(cls, path: str) -> "IQDataset":
        import ast
        d = np.load(path, allow_pickle=False)
        meta = ast.literal_eval(str(d["meta"])) if "meta" in d else {}
        return cls(d["x"], d["y"], float(d["sample_rate_hz"]), meta)
