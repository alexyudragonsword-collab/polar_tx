"""polartx — behavioral simulation of digital polar transmitters.

Two architectures over one composable chain (waveform -> CFR -> polar split
-> envelope path | phase path -> digital PA -> metrics):

- Narrowband: ADPLL two-point phase modulation (BLE GFSK, LTE <= 20 MHz).
- Wideband: open-loop DTC phase modulator (WiFi 6/7 <= 320 MHz,
  5G NR <= 200 MHz).

Phase-path engine adapted from ``pll_simulator`` and waveform/metrics/PA
infrastructure from ``PA_DPD`` (see ``polartx.vendor``).
"""

__version__ = "0.1.0"
