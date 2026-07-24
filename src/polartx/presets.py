"""End-to-end transmitter presets: technology-plausible defaults wired
into ready-to-run PolarTX chains (pllsim.presets convention)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .chain import ChainConfig, PolarTX
from .dpa.dpa import DPA, DPAConfig
from .phasemod.adpll_tp import ADPLLTwoPoint
from .phasemod.dtc_openloop import DTCPhaseModulator, DTCPMConfig
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
              settle_cycles: int = 40_000,
              dp_range_hz: float | None = None) -> TxPreset:
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
                       settle_cycles=settle_cycles, dp_range_hz=dp_range_hz)
    dpa_ = DPA(dpa or DPAConfig(n_bits=8))
    tx = PolarTX(chain or ChainConfig(), pm, dpa_)

    def make_waveform(n_bits: int = 800, pattern: str = "prbs",
                      seed: int = 1) -> Waveform:
        return gfsk_ble(n_bits, fref, rate, pattern=pattern, seed=seed)

    return TxPreset(tx=tx, fs_bb=fref, make_waveform=make_waveform)


def bt_edr_adpll(dpsk: str = "8dpsk", **kw) -> TxPreset:
    """BT EDR2/EDR3 polar TX: same ADPLL two-point phase path as BLE, but
    the DPSK payload is NOT constant-envelope (SRRC, PAPR ~2-3 dB) — the
    envelope path and DPA amplitude codes are genuinely exercised.
    dpsk selects the payload ("8dpsk"/"pi4dqpsk"); kw goes to ble_adpll
    (mode= still selects the response/event phase-modulator engine)."""
    p = ble_adpll(**kw)

    def make_waveform(n_syms: int = 800, seed: int = 1) -> Waveform:
        from .waveforms.edr import edr_dpsk
        return edr_dpsk(n_syms, p.fs_bb, mode=dpsk, seed=seed)

    return TxPreset(tx=p.tx, fs_bb=p.fs_bb, make_waveform=make_waveform)


def ble_1m_adpll(**kw) -> TxPreset:
    return ble_adpll(rate=1e6, **kw)


def ble_2m_adpll(**kw) -> TxPreset:
    return ble_adpll(rate=2e6, **kw)


def lte20_adpll(qam: int = 64, *, mode: str = "response",
                dp_gain: float = 1.0, loop_bw: float = 1e6,
                fref: float = 122.88e6, fout: float = 1.95e9,
                dpa: DPAConfig | None = None,
                chain: ChainConfig | None = None,
                dpd: bool = True, oversampling: int = 4,
                dp_range_hz: float | None = None,
                settle_cycles: int = 60_000) -> TxPreset:
    """LTE 20 MHz narrowband polar TX: ADPLL two-point + DPA.

    Baseband = fref = 122.88 MS/s (30.72 M grid x4, 1:1 with the event
    engine).  Cellular-class DCO (-121 dBc/Hz @ 1 MHz on 1.95 GHz, band-1
    uplink), 1 ps TDC, ~1 MHz loop.  The envelope path runs a mildly
    compressive 10-bit DPA; dpd=True pre-links the exact PolarDPD LUTs
    (factory cal) - set False to see the raw DPA nonlinearity.
    Default chain: 5% hole punching, NO phase-slew limit — unlike the
    quasi-constant-envelope EDR case (ex05), OFDM polar phase slews at
    several x the channel BW everywhere (LTE20: P99 = 39 MHz, tail to
    fs/2), so the direct DAC must simply cover it; slew-limiting below
    ~2x BW destroys the signal.  Use dp_range_hz to study the tradeoff."""
    alpha, rho = design_adpll_dlf(fref, loop_bw, 60.0)
    osc = OscConfig(f0=fout, gain=60e3, pn_dbchz=-121.0, pn_foffset=1e6,
                    pn_f1f3=300e3, pn_floor_dbchz=-155.0)
    pll = ADPLL(ADPLLConfig(
        fref=fref, fout=fout, osc=osc, dlf=DLFConfig(alpha=alpha, rho=rho),
        mode="tdc", tdc=TDCConfig(t_res=1e-12),
        int_band=(1e3, fref / 2)))
    pm = ADPLLTwoPoint(pll, dp_gain=dp_gain, mode=mode,
                       settle_cycles=settle_cycles, dp_range_hz=dp_range_hz)
    dpa_ = DPA(dpa or DPAConfig(n_bits=10, n_thermo=6, sigma_cell=0.002,
                                amam=("rapp", 2.5, 1.1),
                                ampm_deg_poly=(0.0, 2.0, 3.0)))
    dpd_ = None
    if dpd:
        from .cal.polar_dpd import PolarDPD
        dpd_ = PolarDPD.from_dpa(dpa_)
    tx = PolarTX(chain or ChainConfig(env_floor=0.05), pm, dpa_, dpd=dpd_)

    def make_waveform(n_symbols: int = 28, seed: int = 0) -> Waveform:
        from .waveforms.ofdm import lte_waveform
        return lte_waveform(20e6, qam, n_symbols=n_symbols,
                            oversampling=oversampling, seed=seed)

    return TxPreset(tx=tx, fs_bb=fref, make_waveform=make_waveform)


def wifi_dtc(bw: float = 160e6, qam: int = 1024, *, n_bits: int = 11,
             range_ui: float = 1.0, fout: float = 5.9e9,
             oversampling: int = 4, dither: bool = True,
             jitter_rms_s: float = 50e-15, lo_pn: OscConfig | None = None,
             lo_loop_bw: float = 400e3, inl_poly: tuple = (),
             inl_sin: tuple = (),
             cfr_papr_db: float | None = 8.5, env_floor: float = 0.02,
             env_skew_s: float = 0.0, dpa: DPAConfig | None = None
             ) -> TxPreset:
    """WiFi 6/7 wideband polar TX: open-loop DTC phase modulator + DPA.

    Baseband at bw x oversampling (up to 1.28 GS/s at 320 MHz); the DTC
    covers one carrier UI with n_bits resolution.  Defaults: 11-bit
    dithered DTC, 50 fs jitter, WiFi-class fixed LO (-108 dBc/Hz @ 1 MHz
    on 5.9 GHz), CFR to 8.5 dB PAPR, 2% envelope hole punching, 10-bit
    DPA with mild Rapp compression.
    """
    if lo_pn is None:
        # WiFi-7-class synthesizer: -115 dBc/Hz @ 1 MHz on ~6 GHz,
        # ~3 mrad integrated inside a 400 kHz loop
        lo_pn = OscConfig(f0=fout, gain=1.0, pn_dbchz=-115.0,
                          pn_foffset=1e6, pn_f1f3=200e3,
                          pn_floor_dbchz=-155.0)
    pm = DTCPhaseModulator(DTCPMConfig(
        n_bits=n_bits, range_ui=range_ui, dither=dither,
        jitter_rms_s=jitter_rms_s, fout=fout, lo_pn=lo_pn,
        lo_loop_bw=lo_loop_bw, inl_poly=inl_poly, inl_sin=inl_sin))
    dpa_ = DPA(dpa or DPAConfig(n_bits=10, n_thermo=6, sigma_cell=0.002))
    tx = PolarTX(ChainConfig(cfr_papr_db=cfr_papr_db, env_floor=env_floor,
                             env_skew_s=env_skew_s), pm, dpa_)
    fs_bb = bw * oversampling

    def make_waveform(n_symbols: int = 8, seed: int = 0) -> Waveform:
        from .waveforms.ofdm import wifi_waveform
        return wifi_waveform(bw, qam, n_symbols=n_symbols,
                             oversampling=oversampling, seed=seed)

    return TxPreset(tx=tx, fs_bb=fs_bb, make_waveform=make_waveform)


def nr_dtc(bw: float = 100e6, *, scs: float | None = None,
           qam: int | None = None, fout: float | None = None,
           lo_pn: OscConfig | None = None, **kw) -> TxPreset:
    """5G NR wideband polar TX: open-loop DTC + DPA (wifi_dtc machinery).

    Defaults by band: 100 MHz -> FR1 n78 (3.5 GHz, 30 kHz SCS, 256-QAM,
    WiFi-7-class LO); 200 MHz -> FR2 n257 (28 GHz, 120 kHz SCS, 64-QAM,
    -100 dBc/Hz @ 1 MHz mmWave LO).  NR EVM limits: 256-QAM 3.5%
    (-29 dB), 64-QAM 8% (-22 dB)."""
    fr2 = bw > 100e6
    scs = scs or (120e3 if fr2 else 30e3)
    qam = qam or (64 if fr2 else 256)
    fout = fout or (28e9 if fr2 else 3.5e9)
    if lo_pn is None and fr2:
        lo_pn = OscConfig(f0=fout, gain=1.0, pn_dbchz=-100.0,
                          pn_foffset=1e6, pn_f1f3=300e3,
                          pn_floor_dbchz=-140.0)
    p = wifi_dtc(bw=bw, qam=qam, fout=fout, lo_pn=lo_pn, **kw)

    def make_waveform(n_symbols: int = 8, seed: int = 0) -> Waveform:
        from .waveforms.ofdm import nr_waveform
        return nr_waveform(bw, scs, qam, n_symbols=n_symbols, seed=seed)

    return TxPreset(tx=p.tx, fs_bb=p.fs_bb, make_waveform=make_waveform)
