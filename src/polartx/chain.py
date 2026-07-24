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
class SupplyConfig:
    """DPA supply network and LO pushing: the polar-specific AM->PM.

    The class-D DPA draws current ∝ envelope code; through the supply
    impedance (1-pole R / tau decoupling model) that becomes a voltage
    ripple, and LO/DCO supply pushing turns it into phase — an
    envelope-correlated PM with memory that the static polar LUTs
    cannot correct (test-locked; the Cartesian ILA partially can)."""

    r_ohm: float = 0.15            # effective supply impedance
    tau_s: float = 50e-9           # decoupling time constant
    i_fs_a: float = 0.25           # DPA full-scale current draw
    k_push_hz_v: float = 2e6       # LO/DCO pushing [Hz/V]


@dataclass
class ChainConfig:
    env_skew_s: float = 0.0        # AM-path delay relative to PM path (signed)
    cfr_papr_db: float | None = None
    env_floor: float = 0.0         # hole-punch clamp, fraction of rms
    phase_slew_max_hz: float | None = None   # bound phase-path deviation
    phase_interp_win: int = 4      # interp widening around fast runs
    phase_interp: str = "linear"   # "linear" | "smooth" transition shape
    env_headroom: float = 1.0      # full-scale = env_headroom * max envelope
    fs_scale_fixed: float | None = None   # absolute full-scale (envelope
                                   # units): makes the chain a STATIC
                                   # system across runs — required for
                                   # ILA/DPD fitting; None = per-run max
    f_dpa: float | None = None     # DPA amplitude update clock; None = fs_bb
    interleave: int = 1            # staggered DPA banks sharing f_dpa:
                                   # first amplitude image moves to
                                   # interleave * f_dpa (comb-filtered)
    supply: SupplyConfig | None = None   # DPA supply pushing AM->PM


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
        phase-trajectory EVM dict.  DPSK (EDR): differential EVM dict."""
        if self.wf.kind == "ofdm":
            return evm_of_signal(self.y, self.wf.ofdm_ref, equalize=equalize)
        if self.wf.kind == "dpsk":
            from .metrics.dpsk import devm
            return devm(self.y, self.wf)
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

    def avg_efficiency(self, dpa) -> dict:
        """Modulated average drain efficiency of this run's code stream."""
        return dpa.average_efficiency(self.env_code)


class PolarTX:
    def __init__(self, cfg: ChainConfig, phasemod: PhaseModulator, dpa: DPA,
                 dpd=None, memory=None):
        self.cfg = cfg
        self.phasemod = phasemod
        self.dpa = dpa
        self.dpd = dpd                 # PolarDPD or None
        self.memory = memory           # post-DPA memory model: callable y->y

    def run(self, wf: Waveform, *, noise: bool = True, seed: int = 0
            ) -> PolarResult:
        c = self.cfg
        info: dict = {}
        x = wf.x
        if c.cfr_papr_db is not None:
            x = cfr_clip_filter(x, c.cfr_papr_db, wf.fs, wf.bw)
            info["cfr_papr_db"] = c.cfr_papr_db

        env, phase, split_info = polar_split(
            x, c.env_floor, phase_slew_max_hz=c.phase_slew_max_hz,
            fs=wf.fs, phase_interp_win=c.phase_interp_win,
            phase_interp=c.phase_interp)
        info["split"] = split_info

        # envelope path: normalize to DPA full scale, skew, quantize.
        # env_cmd is the ideal DSP-side command (pre-skew) — the reference
        # a skew calibrator correlates against; the skew is an analog
        # path impairment applied on the way to the DPA.
        fs_scale = (c.fs_scale_fixed if c.fs_scale_fixed is not None
                    else c.env_headroom * float(env.max()))
        env_cmd = np.clip(env / fs_scale, 0.0, 1.0)
        env_path = env_cmd
        if self.dpd is not None:
            env_path, ph_corr = self.dpd.predistort(env_cmd)
            phase = phase - ph_corr
            info["dpd"] = True
        if c.env_skew_s:
            env_path = np.clip(
                fractional_delay(env_path, c.env_skew_s * wf.fs), 0.0, 1.0)
        code = self.dpa.encode(env_path)
        codes = [code]
        if c.f_dpa is not None:
            hold = int(round(wf.fs / c.f_dpa))
            if abs(wf.fs / c.f_dpa - hold) > 1e-9:
                raise ValueError("fs/f_dpa must be an integer")
            if hold % c.interleave:
                raise ValueError("hold count must divide by interleave")
            codes = [zoh_hold(code, hold, k * (hold // c.interleave))
                     for k in range(c.interleave)]
            info["dpa_hold"] = hold
            code = codes[0]

        # phase path
        pm = self.phasemod.modulate(phase, wf.fs, noise=noise, seed=seed)
        info["phasemod"] = pm.diagnostics
        phase_tx = pm.phase_out

        # DPA supply pushing: envelope current -> ripple -> LO phase
        if c.supply is not None:
            from scipy.signal import lfilter
            s = c.supply
            i_dpa = s.i_fs_a * self.dpa.amp_table[code]
            a1 = 1.0 / (1.0 + s.tau_s * wf.fs)      # 1-pole IIR
            v = lfilter([a1], [1.0, -(1.0 - a1)], s.r_ohm * i_dpa)
            v = v - v.mean()
            dphi = 2.0 * np.pi * s.k_push_hz_v * np.cumsum(v) / wf.fs
            phase_tx = phase_tx + dphi
            info["supply_phase_rms_mrad"] = float(1e3 * np.std(dphi))

        y = np.mean([self.dpa(ck, phase_tx) for ck in codes],
                    axis=0) * fs_scale
        if self.memory is not None:
            y = self.memory(y)
            info["memory"] = True
        return PolarResult(y=y, fs=wf.fs, wf=wf, env_cmd=env_cmd,
                           env_code=code, phase_cmd=phase,
                           phase_out=pm.phase_out, info=info)
