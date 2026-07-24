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
from .polar import bandwidth_expansion, polar_recombine, polar_split
from .presets import TxPreset, ble_1m_adpll, ble_2m_adpll, ble_adpll, wifi_dtc
from .waveforms import Waveform, gfsk_ble, ofdm_waveform, wifi_waveform

__version__ = "0.1.0"

__all__ = [
    "ChainConfig", "PolarResult", "PolarTX", "DPA", "DPAConfig",
    "PhaseModulator", "PhaseModResult", "IdealPhaseModulator",
    "ADPLLTwoPoint", "DTCPhaseModulator", "DTCPMConfig",
    "polar_split", "polar_recombine", "bandwidth_expansion",
    "TxPreset", "ble_adpll", "ble_1m_adpll", "ble_2m_adpll", "wifi_dtc",
    "Waveform", "gfsk_ble", "ofdm_waveform", "wifi_waveform",
]
