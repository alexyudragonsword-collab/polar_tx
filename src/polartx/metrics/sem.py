"""Spectrum-emission-mask checking with the conventions instruments use.

Two things separate a conformance-style SEM check from comparing a PSD
against a template, and the plain ``check_mask`` path does neither:

1. **Measurement bandwidth.** A SEM limit is power INTEGRATED over a
   specified resolution bandwidth (1 MHz for the LTE general mask,
   100 kHz for BLE in-band, 30 kHz–1 MHz for NR depending on offset), not
   the level of one FFT bin. Comparing raw bins makes the verdict depend
   on ``nfft``.

2. **Which margin you report.** Masks are 0 dBr in-channel and the PSD is
   peak-normalized, so the worst point of a passing signal is *always* the
   tangency at the carrier: the margin reads exactly 0.00 dB no matter how
   much out-of-band headroom there is. Measured on the LTE chain, a run
   with 18.6 dB of true OOB headroom and one with 4.9 dB both report
   0.00 dB. The out-of-band margin is the number a designer actually
   tracks, so it is reported separately.

``MaskSpec`` also carries provenance, so a stylized engineering template
is machine-distinguishable from a spec-sourced table — see ``basis`` and
``source``. The templates shipped here remain stylized; drop in a real
conformance table by constructing a MaskSpec with ``source`` naming the
clause and ``basis="dBm_in_rbw"``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MaskSpec:
    """A spectral mask plus how it is meant to be measured.

    points   : (N, 2) array of (|offset| Hz, limit) breakpoints,
               piecewise-linear, symmetric about the carrier.
    rbw_hz   : measurement bandwidth the limit is defined in.  None keeps
               the raw per-bin comparison (the legacy behaviour).
    basis    : "dBr_peak"    - limit relative to the peak PSD (templates,
                               and the 802.11-style relative masks)
               "dBm_in_rbw" - absolute power in rbw_hz; needs the
                               transmit power to evaluate.
    source   : provenance.  "stylized" marks an engineering template that
               is NOT a conformance clause.
    """
    points: np.ndarray
    rbw_hz: float | None = None
    basis: str = "dBr_peak"
    source: str = "stylized"
    channel_bw_hz: float | None = None

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("points must be an (N, 2) array of "
                             "(offset_hz, limit) breakpoints")
        if self.basis not in ("dBr_peak", "dBm_in_rbw"):
            raise ValueError(f"unknown basis {self.basis!r}")

    @property
    def is_conformance(self) -> bool:
        """True only when the table claims a specification source."""
        return self.source != "stylized"

    def at(self, freqs: np.ndarray) -> np.ndarray:
        return np.interp(np.abs(np.asarray(freqs, float)),
                         self.points[:, 0], self.points[:, 1],
                         right=self.points[-1, 1])


def integrate_to_rbw(freqs: np.ndarray, psd_db: np.ndarray,
                     rbw_hz: float) -> np.ndarray:
    """Convert a per-bin PSD [dB] into power in a sliding ``rbw_hz`` window.

    This is what a spectrum analyser reports: the power in its resolution
    bandwidth, not the height of one bin.  Without it the SEM verdict
    moves with the FFT length.
    """
    freqs = np.asarray(freqs, float)
    psd_db = np.asarray(psd_db, float)
    df = float(np.median(np.diff(freqs)))
    n = max(1, int(round(rbw_hz / df)))
    if n <= 1:
        return psd_db
    lin = 10.0 ** (psd_db / 10.0)
    kern = np.ones(n)
    # 'same' keeps the frequency axis; edges see a partial window, which
    # only ever under-reports at the extreme ends of the record
    acc = np.convolve(lin, kern, mode="same")
    return 10.0 * np.log10(np.maximum(acc, 1e-300))


def check_sem(freqs: np.ndarray, psd_db: np.ndarray, spec: MaskSpec, *,
              channel_bw_hz: float | None = None,
              tx_power_dbm: float | None = None) -> dict:
    """Evaluate ``psd_db`` against ``spec``.

    Returns in-band and out-of-band margins separately; ``passes`` is the
    overall verdict.  Positive margin = below the limit.
    """
    freqs = np.asarray(freqs, float)
    psd_db = np.asarray(psd_db, float)
    bw = channel_bw_hz if channel_bw_hz is not None else spec.channel_bw_hz

    meas = psd_db
    if spec.rbw_hz:
        meas = integrate_to_rbw(freqs, psd_db, spec.rbw_hz)
        meas = meas - meas.max()          # re-normalize to the in-band peak
    if spec.basis == "dBm_in_rbw":
        if tx_power_dbm is None:
            raise ValueError("an absolute (dBm_in_rbw) mask needs "
                             "tx_power_dbm to evaluate")
        meas = meas + tx_power_dbm

    limit = spec.at(freqs)
    margin = limit - meas
    out = {"passes": bool(margin.min() >= 0.0),
           "worst_margin_db": float(margin.min()),
           "limit_db": limit,
           "measured_db": meas,
           "rbw_hz": spec.rbw_hz,
           "basis": spec.basis,
           "source": spec.source,
           "is_conformance": spec.is_conformance}
    if bw:
        oob = np.abs(freqs) > bw / 2.0
        ib = ~oob
        if oob.any():
            out["oob_margin_db"] = float(margin[oob].min())
        if ib.any():
            out["in_band_margin_db"] = float(margin[ib].min())
    return out
