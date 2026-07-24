# Vendored from PA_DPD@44f9ee99: src/padpd/metrics/evm.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Error Vector Magnitude.

EVM is computed between received and reference constellation points after
equalization:

- ``equalize="scalar"``: one complex gain for the whole burst (least
  squares). Keeps frequency-dependent (memory) distortion visible in EVM.
- ``equalize="per_tone"``: independent complex gain per subcarrier, like a
  standard receiver with per-tone channel estimation. Hides linear
  filtering, exposes only nonlinear distortion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..waveform.ofdm import OFDMWaveform, demodulate_ofdm


@dataclass
class EVMResult:
    db: float
    percent: float

    def __str__(self) -> str:
        return f"{self.db:.2f} dB ({self.percent:.3f} %)"


def _evm_from_error(err: np.ndarray, ref: np.ndarray) -> EVMResult:
    ratio = np.sqrt((np.abs(err) ** 2).mean() / (np.abs(ref) ** 2).mean())
    return EVMResult(db=float(20 * np.log10(ratio)), percent=float(100 * ratio))


def evm(rx: np.ndarray, tx: np.ndarray, equalize: str = "scalar") -> EVMResult:
    """EVM between received and transmitted constellation points.

    ``rx``/``tx`` are (n_symbols, n_tones) or flat arrays of complex points.
    """
    rx = np.asarray(rx)
    tx = np.asarray(tx)
    if equalize == "scalar":
        g = np.vdot(tx, rx) / np.vdot(tx, tx)
        err = rx / g - tx
    elif equalize == "per_tone":
        if rx.ndim != 2:
            raise ValueError("per_tone equalization needs (n_symbols, n_tones)")
        g = (np.conj(tx) * rx).sum(axis=0) / (np.abs(tx) ** 2).sum(axis=0)
        err = rx / g - tx
    else:
        raise ValueError(f"unknown equalize mode: {equalize}")
    return _evm_from_error(err, tx)


def evm_of_signal(y: np.ndarray, ref: OFDMWaveform,
                  equalize: str = "scalar") -> EVMResult:
    """Demodulate a time-domain signal against ``ref`` and compute EVM."""
    rx = demodulate_ofdm(y, ref)
    return evm(rx, ref.tx_symbols, equalize=equalize)
