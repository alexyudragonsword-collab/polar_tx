"""Segmented DPA unit-cell array: per-code amplitude with mismatch.

The amplitude DAC is split thermometer MSBs + binary LSBs (standard
SCPA/RF-DAC practice).  Each unit cell has relative random mismatch
sigma_cell; a composite cell of w units gets sigma_cell*sqrt(w) absolute
sigma (units), so binary MSB cells are proportionally better matched but
un-recombined — the classic thermo-vs-binary DNL contrast falls out.  A
systematic linear gradient across the thermometer array models layout
IR-drop/process tilt.
"""
from __future__ import annotations

import numpy as np


def code_amplitude_table(n_bits: int, n_thermo: int, sigma_cell: float,
                         gradient: float, rng: np.random.Generator
                         ) -> np.ndarray:
    """Amplitude [units] for every code 0..2^n_bits-1.

    n_thermo MSBs are thermometer-decoded (2^n_thermo - 1 segments of
    2^(n_bits-n_thermo) units each); the remaining LSBs are binary cells
    of 1, 2, 4, ... units.
    """
    if not 0 <= n_thermo <= n_bits:
        raise ValueError("need 0 <= n_thermo <= n_bits")
    n_bin = n_bits - n_thermo
    seg_w = 1 << n_bin                       # units per thermometer segment
    n_seg = (1 << n_thermo) - 1

    seg = seg_w * np.ones(n_seg)
    if n_seg:
        seg += sigma_cell * np.sqrt(seg_w) * rng.standard_normal(n_seg)
        if n_seg > 1:
            seg *= 1.0 + gradient * (np.arange(n_seg) / (n_seg - 1) - 0.5)
    bin_w = 2.0 ** np.arange(n_bin)          # 1, 2, 4, ... units
    bin_w = bin_w + sigma_cell * np.sqrt(bin_w) * rng.standard_normal(n_bin)

    codes = np.arange(1 << n_bits)
    t_cnt = codes >> n_bin                   # thermometer segments enabled
    seg_cum = np.concatenate(([0.0], np.cumsum(seg)))
    amp = seg_cum[t_cnt]
    for k in range(n_bin):
        amp = amp + np.where((codes >> k) & 1, bin_w[k], 0.0)
    return amp


def inl_dnl(amp: np.ndarray) -> dict:
    """INL/DNL in LSB against the endpoint-fit line."""
    codes = np.arange(amp.size, dtype=float)
    lsb = (amp[-1] - amp[0]) / (amp.size - 1)
    inl = (amp - (amp[0] + lsb * codes)) / lsb
    dnl = np.diff(amp) / lsb - 1.0
    return {"inl_lsb": inl, "dnl_lsb": dnl,
            "inl_max": float(np.max(np.abs(inl))),
            "dnl_max": float(np.max(np.abs(dnl)))}
