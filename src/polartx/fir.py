"""Mixed-domain 2-tap FIR filtering in a polar DTX (Borokhovich, Socher,
Degani, RFIC 2026).

Two DPA arrays (taps), each a full polar chain with its own DTC phase
modulator, are fed the SAME amplitude/phase codes with a programmable
integer delay D between them and coherently combined in a transformer.
The combined transfer is a 2-tap FIR

    h[n] = delta[n] + delta[n-D]  ->  H = 1 + exp(-j*omega*D),
    |H|^2 = 4 cos^2(pi f D / f_s),

so the intended (correlated) content gets +6 dB at DC (the signal, at
0 offset) and deep NOTCHES at f_notch = f_s*(2m+1)/(2D).  The delay D is
in samples of the RF code clock f_s ~ f_0 (~6 GHz), so the first notch
lands hundreds of MHz out — right where the co-located MLO receiver
sits.  The point is coexistence: null the transmitter's out-of-channel
(OOC) quantization/nonlinearity noise floor at the RX offset.

Behavioral model: the two taps share the same deterministic path
(codes -> quantization -> INL -> DPD residual), so that content combines
as x[k] + x[k-D] (FIR-shaped, notched); each tap's RANDOM noise (LO
phase noise, jitter) is independent, so it power-sums and is NOT
notched — exactly the mechanism that turns the -135 dBc/Hz floor into
-155 dBc/Hz at the programmed offset.  Random-noise decorrelation
between taps is what makes the notch a real OOC-noise suppressor rather
than just a signal filter.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .chain import PolarResult, PolarTX
from .impairments import fractional_delay
from .waveforms.base import Waveform


def fir_response(f: np.ndarray, tau_s: float) -> np.ndarray:
    """2-tap FIR H(f) = 1 + exp(-j2*pi*f*tau) for tap delay tau_s."""
    return 1.0 + np.exp(-2j * np.pi * np.asarray(f, float) * tau_s)


def notch_offsets(tau_s: float, f_max: float) -> list[float]:
    """FIR notch offsets |f| = (2m+1)/(2*tau) below f_max."""
    out, m = [], 0
    while (2 * m + 1) / (2.0 * tau_s) < f_max:
        out.append((2 * m + 1) / (2.0 * tau_s))
        m += 1
    return out


def delay_for_notch(notch_offset_hz: float) -> float:
    """Tap delay tau placing the first FIR notch at notch_offset_hz."""
    return 1.0 / (2.0 * notch_offset_hz)


@dataclass
class FIRResult:
    y: np.ndarray
    fs: float
    wf: Waveform
    notch_offset_hz: float
    tau_s: float
    taps: tuple            # (PolarResult tap1, PolarResult tap2)

    def _as_polar(self):
        """Tap 1's PolarResult carrying the COMBINED output.

        Every per-run tap (env_code, phase, waveform) is identical between
        the two taps — only the noise draw and the tap-2 delay differ — so
        substituting the combined y gives the metric layer a result object
        that is correct for the combined signal.  This is what lets the
        dual-tap chain reuse the ordinary PolarResult metrics."""
        return replace(self.taps[0], y=self.y)

    def evm(self, equalize: str = "per_tone", **kw):
        # the 2-tap combine adds a known group delay + in-band tilt that
        # a real receiver equalizes, so score EVM per-tone by default
        return self._as_polar().evm(equalize=equalize, **kw)

    def aclr(self, *a, **kw):
        return self._as_polar().aclr(*a, **kw)

    def check_mask(self, *a, **kw):
        return self._as_polar().check_mask(*a, **kw)

    def avg_efficiency(self, dpa):
        """Combined-chain average efficiency.  Both cores burn DC, so the
        two taps' consumption adds while the combiner sums their voltages
        — reported per-core (the single-core number the DPA model gives),
        which is the meaningful efficiency of each identical tap."""
        return self._as_polar().avg_efficiency(dpa)

    def psd(self, nfft: int = 8192):
        from .vendor.padpd.metrics import psd
        return psd(self.y, self.fs, nfft=nfft)


class FIRDualTapTX:
    """Two-tap mixed-domain FIR polar TX wrapping a base PolarTX.

    notch_offset_hz sets where the first FIR notch lands (the RX-band
    offset for MLO).  run() executes the base chain twice with
    independent noise, delays tap 2 by tau = 1/(2*notch_offset), and
    coherently combines — reproducing the OOC-noise notch."""

    def __init__(self, tx: PolarTX, notch_offset_hz: float = 500e6):
        self.tx = tx
        self.notch_offset_hz = notch_offset_hz

    def run(self, wf: Waveform, *, noise: bool = True, seed: int = 0
            ) -> FIRResult:
        tau = delay_for_notch(self.notch_offset_hz)
        r1 = self.tx.run(wf, noise=noise, seed=seed)
        r2 = self.tx.run(wf, noise=noise, seed=seed + 1)
        y2 = fractional_delay(r2.y, tau * wf.fs)
        y = 0.5 * (r1.y + y2)
        return FIRResult(y=y, fs=wf.fs, wf=wf,
                         notch_offset_hz=self.notch_offset_hz, tau_s=tau,
                         taps=(r1, r2))


def ooc_noise_suppression_db(res_fir: FIRResult, res_single: PolarResult,
                             band_hz: tuple[float, float]) -> float:
    """Measured OOC-noise suppression: mean PSD of the single-tap chain
    minus the FIR-combined chain over a band around the notch [dB]."""
    from .vendor.padpd.metrics import psd
    f1, p1 = psd(res_single.y, res_single.fs, nfft=1 << 14)
    f2, p2 = psd(res_fir.y, res_fir.fs, nfft=1 << 14)
    lo, hi = band_hz
    m = (np.abs(f1) >= lo) & (np.abs(f1) < hi)
    # PSDs are peak-normalized; compare the noise-floor means in-band
    return float(p1[m].mean() - p2[m].mean())
