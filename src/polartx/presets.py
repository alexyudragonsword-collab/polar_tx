"""End-to-end transmitter presets: technology-plausible defaults wired
into ready-to-run PolarTX chains (pllsim.presets convention)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .chain import ChainConfig, PolarTX
from .dpa.dpa import DPA, DPAConfig
from .phasemod.adpll_tp import ADPLLTwoPoint
from .vendor.pllsim.arch.adpll import ADPLL, ADPLLConfig, DLFConfig
from .vendor.pllsim.blocks.oscillator import OscConfig
from .vendor.pllsim.blocks.tdc import TDCConfig
from .vendor.pllsim.synth import design_adpll_dlf
from .waveforms.base import Waveform
from .waveforms.ble import gfsk_ble


@dataclass
class TxPreset:
    tx: PolarTX
    fs_bb: float
    make_waveform: Callable[..., Waveform]


def ble_adpll(rate: float = 1e6, *, mode: str = "response",
              dp_gain: float = 1.0, kdco_est_error: float = 0.0,
              loop_bw: float = 100e3, fref: float = 32e6,
              fout: float = 2.440e9, dpa: DPAConfig | None = None,
              chain: ChainConfig | None = None,
              settle_cycles: int = 40_000) -> TxPreset:
    """BLE LE-1M/LE-2M polar TX: ADPLL two-point PM + DPA at fixed code.

    Baseband runs on the fref grid (fs_bb = fref, 32/16 samples per
    symbol) so event mode needs no resampling.  BLE-class DCO
    (-112 dBc/Hz @ 1 MHz on 2.44 GHz), 10 ps TDC, loop BW ~100 kHz.
    """
    alpha, rho = design_adpll_dlf(fref, loop_bw, 60.0)
    osc = OscConfig(f0=fout, gain=20e3, pn_dbchz=-112.0, pn_foffset=1e6,
                    pn_f1f3=200e3, pn_floor_dbchz=-150.0)
    pll = ADPLL(ADPLLConfig(
        fref=fref, fout=fout, osc=osc, dlf=DLFConfig(alpha=alpha, rho=rho),
        mode="tdc", tdc=TDCConfig(t_res=10e-12),
        kdco_est_error=kdco_est_error, int_band=(1e3, fref / 2)))
    pm = ADPLLTwoPoint(pll, dp_gain=dp_gain, mode=mode,
                       settle_cycles=settle_cycles)
    dpa_ = DPA(dpa or DPAConfig(n_bits=8))
    tx = PolarTX(chain or ChainConfig(), pm, dpa_)

    def make_waveform(n_bits: int = 800, pattern: str = "prbs",
                      seed: int = 1) -> Waveform:
        return gfsk_ble(n_bits, fref, rate, pattern=pattern, seed=seed)

    return TxPreset(tx=tx, fs_bb=fref, make_waveform=make_waveform)


def ble_1m_adpll(**kw) -> TxPreset:
    return ble_adpll(rate=1e6, **kw)


def ble_2m_adpll(**kw) -> TxPreset:
    return ble_adpll(rate=2e6, **kw)
