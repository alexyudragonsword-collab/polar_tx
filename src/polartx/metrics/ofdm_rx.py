"""Standard-receiver-style OFDM EVM: pilot CPE tracking and
preamble-based channel estimation.

The plain padpd EVM equalizes with one LS gain (or data-directed
per-tone gains) over the whole burst.  A real receiver estimates the
channel from known PREAMBLE symbols and tracks per-symbol common phase
from PILOT tones — measuring EVM over the data tones only.  Channel
coding is deliberately absent everywhere in polartx: nothing a TX
impairment does is measured downstream of the bit mapping.
"""
from __future__ import annotations

import numpy as np

from ..vendor.padpd.metrics.evm import EVMResult, _evm_from_error
from ..waveforms.base import Waveform
from ..waveforms.ofdm import demodulate_ofdm


def evm_rx(y: np.ndarray, wf: Waveform, *, track_cpe: bool = True
           ) -> EVMResult:
    """EVM with the receiver features the waveform carries.

    - preamble_symbols > 0: per-tone channel estimate from the known
      training rows (a real receiver's equalizer), applied to the rest.
    - n_pilots > 0 and track_cpe: per-symbol common-phase correction
      from the pilot tones before the error is scored.
    EVM is scored over DATA tones of DATA symbols only.
    """
    rx = demodulate_ofdm(y, wf.ofdm_ref)
    tx = wf.ofdm_ref.tx_symbols
    n_pre = wf.meta.get("preamble_symbols", 0)
    pilot_idx = np.asarray(wf.meta.get("pilot_idx", []), dtype=int)

    if n_pre > 0:
        h_hat = np.mean(rx[:n_pre] / tx[:n_pre], axis=0)   # per-tone LS
        rx = rx / h_hat
    rx_d, tx_d = rx[n_pre:], tx[n_pre:]

    if pilot_idx.size and track_cpe:
        cpe = np.angle(np.sum(rx_d[:, pilot_idx]
                              * np.conj(tx_d[:, pilot_idx]),
                              axis=1, keepdims=True))
        rx_d = rx_d * np.exp(-1j * cpe)

    data_mask = np.ones(tx_d.shape[1], dtype=bool)
    if pilot_idx.size:
        data_mask[pilot_idx] = False
    rx_d, tx_d = rx_d[:, data_mask], tx_d[:, data_mask]
    if n_pre == 0:
        g = np.vdot(tx_d, rx_d) / np.vdot(tx_d, tx_d)      # scalar LS
        rx_d = rx_d / g
    return _evm_from_error(rx_d - tx_d, tx_d)
