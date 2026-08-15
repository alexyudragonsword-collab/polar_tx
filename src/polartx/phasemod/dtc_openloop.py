"""Wideband phase modulator: open-loop DTC / phase interpolator.

A clean fixed-frequency LO (from an integer PLL, not modulated) is
phase-shifted sample-by-sample by a DTC covering range_ui carrier UI
(2*pi*range_ui rad, wrapping modulo that range).  Everything runs on the
baseband oversampled grid, so multi-hundred-MHz modulation is just a
vectorized table map — this is what makes the architecture wideband.

Modeled impairments: phase quantization (n_bits over the range, optional
first-order error-feedback dither), gain error, INL (polynomial + sine,
in UI — pllsim DTC conventions), random DTC jitter, LO phase noise
(Leeson via colored synthesis), and a phase-update clock slower than the
baseband grid (ZOH -> modulation images at multiples of f_update).
Deterministic imperfections (quantization/INL/gain) are always applied;
noise=False gates only the random terms, matching the pllsim engines.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..impairments import zoh_hold
from ..vendor.pllsim.blocks.oscillator import OscConfig
from ..vendor.pllsim.core.colored import synth_from_psd
from .base import PhaseModResult, PhaseModulator

TWOPI = 2.0 * np.pi


@dataclass
class DTCPMConfig:
    """Open-loop DTC phase modulator parameters.

    Sizing guide, from the design sweeps in ``examples/ex03``:

    n_bits
        Sets the phase-quantization EVM floor, ``(2*pi/2**B)/sqrt(12)``
        spread over the oversampling ratio.  1024-QAM wants ~9 bit and
        4096-QAM ~10 bit over a 1 UI range; ``dither=True`` buys about
        one bit in band by first-order shaping the quantization error
        out of the channel.
    range_ui
        DTC span in carrier UI.  1 UI is the minimum that can cover an
        arbitrary phase (the command is wrapped modulo the range), and
        widening it costs resolution at fixed n_bits.
    inl_poly / inl_sin
        Static nonlinearity in UI (pllsim conventions).  The sine term is
        the one that matters for spectrum: amplitude ``A`` UI at
        ``cycles`` per full range puts spurs at multiples of the
        modulation offset at roughly ``20*log10(pi*A)`` dBc.
    jitter_rms_s
        Random edge jitter, converted to rad through ``fout``: a white
        phase floor that no calibration removes.
    f_update
        Phase update clock when slower than the baseband grid.  ZOH puts
        modulation images at multiples of f_update with sinc rolloff —
        an aliasing budget, not an impairment to calibrate away.
    lo_pn / lo_loop_bw
        The fixed LO is PLL-locked, so its Leeson PSD is flattened below
        the loop bandwidth.  This is why CPE is negligible in these
        presets: the residual phase noise is high-pass shaped above the
        symbol rate.
    """

    n_bits: int = 10                # phase resolution over the full range
    range_ui: float = 1.0           # DTC span in carrier UI (2*pi*range_ui rad)
    inl_poly: tuple = ()            # polynomial in code/fullscale -> UI
    inl_sin: tuple = ()             # (amp_ui, cycles, phase_rad)
    gain_error: float = 0.0         # fractional range/gain error
    jitter_rms_s: float = 0.0       # random edge jitter [s]
    dither: bool = False            # 1st-order error feedback on the code
    f_update: float | None = None   # phase update clock; None = fs_bb
    fout: float = 6e9               # carrier (converts s <-> rad)
    lo_pn: OscConfig | None = None  # fixed-LO Leeson phase noise
    lo_loop_bw: float = 200e3       # LO is PLL-locked: PSD flattened below
                                    # the loop BW (in-band suppression)

    @property
    def range_rad(self) -> float:
        """Full DTC span in radians."""
        return TWOPI * self.range_ui

    @property
    def lsb_rad(self) -> float:
        """Phase step of one code, ``range_rad / 2**n_bits``."""
        return self.range_rad / (1 << self.n_bits)


def _efm1_quantize(x: np.ndarray) -> np.ndarray:
    """First-order error-feedback quantizer, vectorized:
    y[n] = floor(x[n] + e[n-1]) == diff(floor(cumsum(x)))."""
    c = np.floor(np.cumsum(x))
    return np.diff(np.concatenate(([0.0], c)))


class DTCPhaseModulator(PhaseModulator):
    """The wideband transmitter's phase path (see the module docstring).

    No loop, so no loop bandwidth to trade against modulation bandwidth:
    the whole trajectory is a vectorized table map on the baseband grid,
    which is what lets this cover 320 MHz where a two-point ADPLL cannot.
    The price is that every DTC imperfection lands directly on the
    carrier, unattenuated by any loop — hence the calibration hooks.

    ``cal.dtc_cal.apply_dtc_correction`` fills ``gain_hat`` and the
    ``inl_lut_*`` pair from CW training; both are pre-subtracted from the
    command here, so a calibrated modulator is the same object with state
    set, not a different class.
    """

    def __init__(self, cfg: DTCPMConfig):
        self.cfg = cfg
        # calibration state (set by cal.dtc_cal.apply_dtc_correction):
        # estimated gain error and a per-segment INL LUT [rad] vs
        # normalized code, pre-subtracted from the command
        self.gain_hat = 0.0
        self.inl_lut_rad: np.ndarray | None = None
        self.inl_lut_x: np.ndarray | None = None

    def _inl_rad(self, code: np.ndarray) -> np.ndarray:
        c = self.cfg
        xn = code / (1 << c.n_bits)              # normalized code [0,1)
        t = np.zeros_like(xn)
        if c.inl_poly:
            t = t + np.polyval(tuple(c.inl_poly)[::-1], xn)
        if c.inl_sin:
            amp, cyc, ph = c.inl_sin
            t = t + amp * np.sin(TWOPI * cyc * xn + ph)
        return TWOPI * t                          # UI -> rad

    def modulate(self, phase_cmd, fs_bb, *, noise=True, seed=0):
        """Transmit ``phase_cmd`` [rad] @ ``fs_bb`` through the DTC.

        Order of operations mirrors the hardware: calibration
        pre-correction, wrap modulo the range, quantize (optionally
        dithered), apply gain error and INL, ZOH to ``f_update``, then add
        jitter and LO phase noise.  ``noise=False`` gates only the last
        two."""
        c = self.cfg
        phase_cmd = np.asarray(phase_cmd, dtype=float)
        rng = np.random.default_rng(seed)
        m = c.range_rad
        ph_w = np.mod(phase_cmd, m)

        target = ph_w
        if self.inl_lut_rad is not None:
            target = target - np.interp(ph_w / m, self.inl_lut_x,
                                        self.inl_lut_rad)
        if self.gain_hat != 0.0:
            target = target / (1.0 + self.gain_hat)
        code_ideal = target / c.lsb_rad
        if c.dither:
            code = _efm1_quantize(code_ideal)
        else:
            code = np.rint(code_ideal)
        code = np.mod(code, 1 << c.n_bits)

        ph_q = code * c.lsb_rad * (1.0 + c.gain_error) + self._inl_rad(code)
        # rejoin across wrap events: keep the (small) wrapped error only
        err = np.mod(ph_q - ph_w + 0.5 * m, m) - 0.5 * m
        out = phase_cmd + err

        if c.f_update is not None:
            ratio = fs_bb / c.f_update
            hold = int(round(ratio))
            if abs(ratio - hold) > 1e-9:
                raise ValueError("fs_bb/f_update must be an integer")
            out = zoh_hold(out, hold)

        diag = {"mode": "dtc_openloop", "code": code,
                "lsb_rad": c.lsb_rad}
        if noise:
            if c.jitter_rms_s > 0.0:
                out = out + rng.normal(0.0, TWOPI * c.fout * c.jitter_rms_s,
                                       out.size)
            if c.lo_pn is not None:
                src = c.lo_pn.leeson("lo")
                # locked-LO approximation: inside the PLL loop BW the
                # oscillator's f^-2/f^-3 slopes are flattened
                pn = synth_from_psd(
                    lambda f: src.psd(np.maximum(f, c.lo_loop_bw)),
                    fs_bb, out.size, rng)
                out = out + pn
                diag["lo_pn"] = True
        return PhaseModResult(phase_out=out, diagnostics=diag)
