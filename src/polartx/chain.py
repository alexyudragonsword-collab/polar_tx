"""The composable polar TX chain shared by both flavors.

Waveform -> [CFR] -> polar split -> envelope path (normalize, skew, DPA
amplitude code) | phase path (PhaseModulator) -> DPA recombine -> metrics.
Swap the PhaseModulator to move between the narrowband (ADPLL two-point)
and wideband (open-loop DTC) transmitters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dpa.dpa import DPA
from .impairments import fractional_delay, zoh_hold
from .phasemod.base import PhaseModulator
from .polar.split import polar_split
from .vendor.padpd.cfr import cfr_clip_filter
from .vendor.padpd.metrics import (aclr, check_mask, evm_of_signal, psd)
from .waveforms.base import Waveform


@dataclass
class ChainConfig:
    env_skew_s: float = 0.0        # AM-path delay relative to PM path (signed)
    cfr_papr_db: float | None = None
    env_floor: float = 0.0         # hole-punch clamp, fraction of rms
    env_headroom: float = 1.0      # full-scale = env_headroom * max envelope
    f_dpa: float | None = None     # DPA amplitude update clock; None = fs_bb


@dataclass
class PolarResult:
    y: np.ndarray                  # chain output, complex baseband @ fs
    fs: float
    wf: Waveform
    env_cmd: np.ndarray            # normalized envelope command [0,1]
    env_code: np.ndarray           # DPA amplitude codes
    phase_cmd: np.ndarray          # commanded phase [rad]
    phase_out: np.ndarray          # phase-modulator output [rad]
    info: dict = field(default_factory=dict)

    # ------------------------------------------------------------ metrics
    def evm(self, equalize: str = "scalar"):
        """OFDM: constellation EVM vs the reference grid.  GFSK:
        phase-trajectory EVM dict (vendored pllsim convention)."""
        if self.wf.kind == "ofdm":
            return evm_of_signal(self.y, self.wf.ofdm_ref, equalize=equalize)
        from .metrics.ble_metrics import phase_evm
        return phase_evm(self.y, self.wf)

    def aclr(self):
        return aclr(self.y, self.fs, self.wf.bw)

    def psd(self, nfft: int = 4096):
        return psd(self.y, self.fs, nfft=nfft)

    def check_mask(self, mask=None, nfft: int = 4096):
        from .metrics.masks import default_mask
        if mask is None:
            mask = default_mask(self.wf)
        f, p = self.psd(nfft=nfft)
        return check_mask(f, p, mask)


class PolarTX:
    def __init__(self, cfg: ChainConfig, phasemod: PhaseModulator, dpa: DPA):
        self.cfg = cfg
        self.phasemod = phasemod
        self.dpa = dpa

    def run(self, wf: Waveform, *, noise: bool = True, seed: int = 0
            ) -> PolarResult:
        c = self.cfg
        info: dict = {}
        x = wf.x
        if c.cfr_papr_db is not None:
            x = cfr_clip_filter(x, c.cfr_papr_db, wf.fs, wf.bw)
            info["cfr_papr_db"] = c.cfr_papr_db

        env, phase, split_info = polar_split(x, c.env_floor)
        info["split"] = split_info

        # envelope path: normalize to DPA full scale, skew, quantize.
        # env_cmd is the ideal DSP-side command (pre-skew) — the reference
        # a skew calibrator correlates against; the skew is an analog
        # path impairment applied on the way to the DPA.
        fs_scale = c.env_headroom * float(env.max())
        env_cmd = env / fs_scale
        env_path = env_cmd
        if c.env_skew_s:
            env_path = np.clip(
                fractional_delay(env_cmd, c.env_skew_s * wf.fs), 0.0, 1.0)
        code = self.dpa.encode(env_path)
        if c.f_dpa is not None:
            hold = int(round(wf.fs / c.f_dpa))
            if abs(wf.fs / c.f_dpa - hold) > 1e-9:
                raise ValueError("fs/f_dpa must be an integer")
            code = zoh_hold(code, hold)
            info["dpa_hold"] = hold

        # phase path
        pm = self.phasemod.modulate(phase, wf.fs, noise=noise, seed=seed)
        info["phasemod"] = pm.diagnostics

        y = self.dpa(code, pm.phase_out) * fs_scale
        return PolarResult(y=y, fs=wf.fs, wf=wf, env_cmd=env_cmd,
                           env_code=code, phase_cmd=phase,
                           phase_out=pm.phase_out, info=info)
