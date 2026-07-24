"""polartx — behavioral simulation of digital polar transmitters.

Two architectures over one composable chain (waveform -> CFR -> polar
split -> envelope path | phase path -> digital PA -> metrics):

- Narrowband: ADPLL two-point phase modulation (BLE GFSK, LTE <= 20 MHz).
- Wideband: open-loop DTC phase modulator (WiFi 6/7 <= 320 MHz,
  5G NR <= 200 MHz).

Phase-path engine adapted from ``pll_simulator`` and waveform/metrics/PA
infrastructure from ``PA_DPD`` (see ``polartx.vendor``).
"""

from .chain import ChainConfig, PolarResult, PolarTX
from .dpa import DPA, DPAConfig
from .phasemod import (ADPLLTwoPoint, DTCPhaseModulator, DTCPMConfig,
                       IdealPhaseModulator, PhaseModResult, PhaseModulator)
from .cal.polar_dpd import PolarDPD
from .montecarlo import MCResult, run_mc, wifi_chip_builder
from .polar import bandwidth_expansion, polar_recombine, polar_split
from .presets import (TxPreset, bench_edge_polar_staszewski05,
                      bench_lte20_polar_madoglio14, bench_wifi11n_polar,
                      ble_1m_adpll, ble_2m_adpll, ble_adpll, bt_edr_adpll,
                      lte20_adpll, nr_dtc, wifi_dtc)
from .waveforms import (Waveform, edr_dpsk, gfsk_ble, ofdm_waveform,
                        wifi_waveform)
from .waveforms.ofdm import lte_waveform, nr_waveform

__version__ = "0.1.0"

__all__ = [
    "ChainConfig", "PolarResult", "PolarTX", "DPA", "DPAConfig",
    "PhaseModulator", "PhaseModResult", "IdealPhaseModulator",
    "ADPLLTwoPoint", "DTCPhaseModulator", "DTCPMConfig",
    "polar_split", "polar_recombine", "bandwidth_expansion",
    "PolarDPD", "MCResult", "run_mc", "wifi_chip_builder",
    "TxPreset", "ble_adpll", "ble_1m_adpll", "ble_2m_adpll", "bt_edr_adpll",
    "lte20_adpll", "nr_dtc", "wifi_dtc",
    "bench_edge_polar_staszewski05", "bench_lte20_polar_madoglio14",
    "bench_wifi11n_polar",
    "Waveform", "edr_dpsk", "gfsk_ble", "lte_waveform", "nr_waveform",
    "ofdm_waveform", "wifi_waveform",
]
