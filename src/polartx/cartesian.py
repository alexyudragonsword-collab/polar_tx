"""Cartesian (digital I/Q) transmitter, the third topology beside polar and
outphasing.

    Waveform -> [CFR] -> I = Re x, Q = Im x -> two signed codes -> RFDAC -> y

No decomposition into envelope and phase, no phase modulator, no combiner:
the constellation is drawn in I/Q directly.  The chain is therefore immune
to everything the polar chain is sharpest about — AM/PM path skew, envelope
hole punching, phase-path bandwidth expansion — and exposed instead to what
an RF-DAC does: I/Q imbalance (image), LO leakage (carrier), clock jitter,
unit-cell mismatch on two arrays, and an efficiency law that pays for
|I| + |Q| while delivering I² + Q².

``CartesianTX`` takes the SAME ``ChainConfig`` the other two topologies
take, so one description drives all three, but it only reads the
architecture-agnostic part of it:

- ``cfr_papr_db`` — applied to the composite exactly as the polar chain
  applies it, so all topologies see the identical waveform.

Everything else in ``ChainConfig`` describes an envelope or a phase path
this chain does not have and is IGNORED: ``env_skew_s``, ``env_floor``,
``env_headroom``, ``fs_scale_fixed``, ``phase_slew_max_hz``,
``phase_interp_win``, ``phase_interp``, ``f_dpa``, ``interleave``,
``supply``.  Ignored is not silent: a non-default value in any of them
raises a ``UserWarning`` naming the field when the chain is built, because
a knob that reads as wired and does nothing is the failure mode this repo
keeps finding (CONTRIBUTING.md 1).

Like ``fir.py`` and ``outphasing.py`` this is a sibling class with its own
result type carrying the same-named metric methods; ``chain.py`` and
``PolarResult`` are untouched.  ``CartesianResult`` scores itself with the
same vendored functions ``PolarResult`` uses rather than through a borrowed
``PolarResult``, because there is no polar run to borrow — the envelope and
phase taps genuinely do not exist here.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields

import numpy as np

from .chain import ChainConfig
from .rfdac import RFDAC
from .vendor.padpd.cfr import cfr_clip_filter
from .vendor.padpd.metrics import aclr, check_mask, evm_of_signal, psd
from .waveforms.base import Waveform

#: ChainConfig fields that describe an envelope or a phase path.  A
#: non-default value in any of these is warned about, not obeyed.
IGNORED_CHAIN_FIELDS = ("env_skew_s", "env_floor", "env_headroom",
                        "fs_scale_fixed", "phase_slew_max_hz",
                        "phase_interp_win", "phase_interp", "f_dpa",
                        "interleave", "supply")


def ignored_chain_settings(cfg: ChainConfig) -> list[str]:
    """Names of the envelope/phase-path fields set away from their defaults
    in ``cfg`` — what ``CartesianTX`` will warn about."""
    defaults = {f.name: getattr(ChainConfig(), f.name) for f in fields(ChainConfig)}
    return [n for n in IGNORED_CHAIN_FIELDS if getattr(cfg, n) != defaults[n]]


@dataclass
class CartesianResult:
    """One Cartesian run: the output plus the two code streams.

    ``i_code`` / ``q_code`` are the signed RF-DAC codes, the analogue of
    ``PolarResult.env_code`` — read them for the realized quantization and
    for the efficiency, which is a function of the code pair.  Scored with
    the same scalar equalization as ``PolarResult`` and ``OutphasingResult``,
    so all three topologies are compared on one convention.
    """

    y: np.ndarray
    fs: float
    wf: Waveform
    i_code: np.ndarray
    q_code: np.ndarray
    rfdac: RFDAC
    info: dict = field(default_factory=dict)

    #: same convention as the other two topologies — no linear response to
    #: equalize away
    evm_equalize_default = "scalar"

    def evm(self, equalize: str = "scalar"):
        """OFDM constellation EVM; DPSK differential EVM; GFSK phase EVM —
        the same dispatch as ``PolarResult.evm``."""
        if self.wf.kind == "ofdm":
            return evm_of_signal(self.y, self.wf.require_ofdm_ref(),
                                 equalize=equalize)
        if self.wf.kind == "dpsk":
            from .metrics.dpsk import devm
            return devm(self.y, self.wf)
        from .metrics.ble_metrics import phase_evm
        return phase_evm(self.y, self.wf)

    def aclr(self):
        """Adjacent-channel leakage vs this waveform's channel bandwidth."""
        return aclr(self.y, self.fs, self.wf.bw)

    def psd(self, nfft: int = 4096):
        """Welch PSD ``(f, p_db)`` of the chain output."""
        return psd(self.y, self.fs, nfft=nfft)

    def check_mask(self, mask=None, nfft: int = 4096):
        """Per-bin comparison against a spectral template (default: this
        waveform's), as ``PolarResult.check_mask``."""
        from .metrics.masks import default_mask
        if mask is None:
            mask = default_mask(self.wf)
        f, p = self.psd(nfft=nfft)
        return check_mask(f, p, mask)

    def avg_efficiency(self, rfdac: RFDAC | None = None) -> dict:
        """Modulated average drain efficiency of this run's code pair under
        the RF-DAC's I² + Q² / (|I| + |Q|) law.  The argument mirrors
        ``PolarResult.avg_efficiency(dpa)``; ``None`` uses this run's DAC."""
        return (rfdac or self.rfdac).average_efficiency(self.i_code, self.q_code)


class CartesianTX:
    """A complete digital I/Q transmitter: chain config + RF-DAC.

    Parameters
    ----------
    cfg : ChainConfig
        Only ``cfr_papr_db`` is read; see the module docstring for what is
        ignored and warned about.
    rfdac : RFDAC
        The two-array output stage.
    """

    def __init__(self, cfg: ChainConfig, rfdac: RFDAC):
        self.cfg = cfg
        self.rfdac = rfdac
        self.ignored = ignored_chain_settings(cfg)
        if self.ignored:
            warnings.warn(
                "CartesianTX has no envelope or phase path; these ChainConfig "
                f"settings are ignored: {', '.join(self.ignored)}",
                UserWarning, stacklevel=2)

    def run(self, wf: Waveform, *, noise: bool = True, seed: int = 0
            ) -> CartesianResult:
        """Run one burst: CFR, per-axis normalization to the DAC full scale,
        quantization, the RF-DAC with its impairments.

        Full scale is the larger of the two axes' peaks, so the array that
        peaks first reaches its top code — the way an I/Q DAC is actually
        driven — and a diagonal peak can exceed an on-axis full scale by
        up to √2 (that headroom is the DAC's, not a clip).  ``noise=False``
        gates the clock jitter only; mismatch, imbalance and leakage are
        deterministic and always apply.
        """
        c = self.cfg
        info: dict = {"ignored": list(self.ignored)}
        x = wf.x
        if c.cfr_papr_db is not None:
            x = cfr_clip_filter(x, c.cfr_papr_db, wf.fs, wf.bw)
            info["cfr_papr_db"] = c.cfr_papr_db
        fs_scale = float(max(np.abs(x.real).max(), np.abs(x.imag).max()))
        i_code = self.rfdac.encode(x.real / fs_scale)
        q_code = self.rfdac.encode(x.imag / fs_scale)
        y = self.rfdac(i_code, q_code, noise=noise, seed=seed, fs=wf.fs) * fs_scale
        info["fs_scale"] = fs_scale
        info["backoff_db"] = self.rfdac.average_efficiency(i_code, q_code)["backoff_db"]
        return CartesianResult(y=y, fs=wf.fs, wf=wf, i_code=i_code,
                               q_code=q_code, rfdac=self.rfdac, info=info)
