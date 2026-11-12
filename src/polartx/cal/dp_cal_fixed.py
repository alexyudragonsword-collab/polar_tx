"""Fixed-point sign-sign LMS for the two-point direct-path gain (bit-true).

The float calibrator the event engine runs (vendored
``pllsim.calibration.lms.SignSignLMS``) keeps four float64 states and
steps by ``mu = 2e-5``.  This is the same update rule with every state an
integer, so a hardware implementation can be checked against it sample
by sample (dsp_ip ``ip/dp_cal_lms``):

* inputs are quantized first: the phase error to ``e_bits`` signed with
  LSB ``2^-e_frac`` UI, the modulation regressor to ``reg_bits`` signed
  with LSB ``reg_lsb_hz`` (round to nearest, then saturate);
* both centering means are leaky integrators ``A += x - (A >>> k)``
  (``A`` is the mean times ``2^k``: ``ema = 2^-k``, float uses 1e-3),
  and the centered sample is compared at full precision,
  ``d = (x << k) - A``, so the sign carries no truncation bias;
* ``sgn`` keeps an exact zero (a zero regressor -- the unmodulated
  settle -- makes no update, exactly as in float);
* the value register ``V`` is unsigned ``v_bits`` wide with LSB
  ``2^-v_frac``: each step is exactly +-1 LSB (``mu = 2^-v_frac``,
  1.53e-5 by default) and saturates instead of wrapping.

``FixedSignSignLMS`` has the float class's object contract
(``.step(err, reg)``, a writable ``.value``) and the engine's kernel
contract (``kernel_view`` / ``kernel_done``: the update runs inside the
ADPLL's per-cycle kernel as ``fixed_dp_step`` over a state vector), so it
drops into ``ADPLLTwoPoint.dp_cal`` unchanged -- with the float class's
limit: the centred regressor needs a zero-mean modulation, which a
frequency ramp is not (see ``ADPLLTwoPoint.__init__``).  ``golden_trace``
is the same update on already-quantized integer codes -- the vector form
that RTL is compared against.  Every path (object, kernel, golden trace)
goes through the same two scalar kernels, ``_quantize`` and ``_update``,
so there is one update rule, not three.  Only numpy is required: inside
polartx the kernels compile when the vendored jit finds numba, and copied
out on its own the file falls back to plain Python, so it can still be
vendored by itself.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

try:
    from ..vendor.pllsim.core.jit import kernel
except ImportError:          # vendored on its own: the plain-Python path
    def kernel(fn):          # type: ignore[misc]
        return fn


@dataclass(frozen=True)
class DpCalFixedConfig:
    """Word widths of the fixed-point calibrator (all are configuration)."""

    e_bits: int = 10          # phase-error input, signed
    e_frac: int = 12          # its LSB is 2^-e_frac UI (+-0.125 UI full scale)
    reg_bits: int = 12        # regressor (modulation frequency) input, signed
    reg_lsb_hz: float = 500.0  # its LSB in Hz (+-1.02 MHz full scale)
    k: int = 10               # centering EMA = 2^-k
    v_bits: int = 18          # value register, unsigned
    v_frac: int = 16          # its LSB = the step size mu = 2^-v_frac

    def __post_init__(self):
        for name in ("e_bits", "reg_bits", "v_bits"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be >= 2")
        if self.k < 1 or self.e_frac < 0 or self.v_frac < 0:
            raise ValueError("k >= 1, e_frac >= 0, v_frac >= 0")
        if not self.reg_lsb_hz > 0:
            raise ValueError("reg_lsb_hz must be > 0")

    @property
    def mu(self) -> float:
        return 2.0 ** -self.v_frac

    @property
    def v_max(self) -> int:
        return (1 << self.v_bits) - 1

    def to_dict(self) -> dict:
        return asdict(self)


@kernel
def _sat(x: int, lo: int, hi: int) -> int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@kernel
def _sgn(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


@kernel
def _quantize(x: float, mul: float, div: float, half: int) -> int:
    """``round(x * mul / div)`` to nearest, ties to even (Python ``round``,
    numpy ``rint``), saturated to a signed code of ``half`` = 2^(bits-1).
    ``x * 1.0`` and ``y / 1.0`` are exact, so one function serves the
    error (a multiply) and the regressor (a divide) bit for bit."""
    return _sat(int(round(x * mul / div)), -half, half - 1)


@kernel
def _update(v: int, a_reg: int, a_err: int, e: int, r: int, k: int,
            v_max: int) -> tuple[int, int, int]:
    """One sign-sign step on integer codes; returns (V, A_reg, A_err)."""
    a_reg += r - (a_reg >> k)              # >> on a signed int is arithmetic
    dr = (r << k) - a_reg
    a_err += e - (a_err >> k)
    de = (e << k) - a_err
    v = _sat(v + _sgn(de) * _sgn(dr), 0, v_max)
    return v, a_reg, a_err


def quantize_err(err: float, cfg: DpCalFixedConfig) -> int:
    """Phase error [UI] -> signed e_bits code (round to nearest, ties to
    even -- Python ``round``, same as numpy ``rint`` -- then saturate)."""
    return int(_quantize(err, float(1 << cfg.e_frac), 1.0, 1 << (cfg.e_bits - 1)))


def quantize_reg(reg: float, cfg: DpCalFixedConfig) -> int:
    """Regressor [Hz] -> signed reg_bits code."""
    return int(_quantize(reg, 1.0, float(cfg.reg_lsb_hz), 1 << (cfg.reg_bits - 1)))


def value_code(value: float, cfg: DpCalFixedConfig) -> int:
    """Gain -> value-register code (round, saturate to [0, v_max])."""
    return _sat(int(round(value * (1 << cfg.v_frac))), 0, cfg.v_max)


def step_codes(state: list[int], e: int, r: int, cfg: DpCalFixedConfig) -> int:
    """One update on integer codes; ``state = [V, A_reg, A_err]`` is
    modified in place and the new ``V`` returned."""
    v, a_reg, a_err = _update(state[0], state[1], state[2], e, r, cfg.k,
                              cfg.v_max)
    state[:] = [int(v), int(a_reg), int(a_err)]
    return state[0]


@kernel
def fixed_dp_step(st: np.ndarray, pr: np.ndarray, err: float, reg: float,
                  rec: np.ndarray, n: int) -> float:
    """The calibrator as the ADPLL kernel's dp_cal step (see
    ``FixedSignSignLMS.kernel_view`` for ``pr``).  Records the input codes
    and the new value code of cycle ``n`` in ``rec[n]``."""
    e = _quantize(err, pr[0], 1.0, int(pr[1]))
    r = _quantize(reg, 1.0, pr[2], int(pr[3]))
    v, a_reg, a_err = _update(int(st[0]), int(st[1]), int(st[2]), e, r,
                              int(pr[4]), int(pr[5]))
    st[0] = v
    st[1] = a_reg
    st[2] = a_err
    rec[n, 0] = e
    rec[n, 1] = r
    rec[n, 2] = v
    return v / pr[6]


def golden_trace(e_codes, r_codes, v0: int,
                 cfg: DpCalFixedConfig = DpCalFixedConfig()) -> np.ndarray:
    """Value-register code after every step, from the input code streams
    (both centering accumulators start at zero, ``V`` at ``v0``)."""
    e_codes = np.asarray(e_codes, dtype=np.int64)
    r_codes = np.asarray(r_codes, dtype=np.int64)
    if e_codes.shape != r_codes.shape or e_codes.ndim != 1:
        raise ValueError("e_codes and r_codes must be 1-D and the same length")
    he, hr = 1 << (cfg.e_bits - 1), 1 << (cfg.reg_bits - 1)
    if e_codes.size and (e_codes.min() < -he or e_codes.max() >= he
                         or r_codes.min() < -hr or r_codes.max() >= hr):
        raise ValueError("input codes exceed e_bits / reg_bits")
    if not 0 <= v0 <= cfg.v_max:
        raise ValueError("v0 outside the value register")
    st = [int(v0), 0, 0]
    out = np.empty(e_codes.size, dtype=np.int64)
    for i, (e, r) in enumerate(zip(e_codes.tolist(), r_codes.tolist())):
        out[i] = step_codes(st, e, r, cfg)
    return out


class FixedSignSignLMS:
    """Bit-true sign-sign LMS gain calibrator (engine contract of
    ``SignSignLMS``: ``.step(err, reg)`` and a writable ``.value``).

    Every call also records the quantized input codes and the new value
    code (``e_codes``, ``r_codes``, ``v_codes``), which is what a golden
    vector set is made of; ``v_start`` is the code the run started from.
    """

    def __init__(self, init: float = 1.0,
                 cfg: DpCalFixedConfig = DpCalFixedConfig()):
        self.cfg = cfg
        self._st = [0, 0, 0]
        self.v_start = 0
        self.value = init
        self.e_codes: list[int] = []
        self.r_codes: list[int] = []
        self.v_codes: list[int] = []

    @property
    def mu(self) -> float:
        return self.cfg.mu

    @property
    def code(self) -> int:
        return self._st[0]

    @property
    def value(self) -> float:
        return self._st[0] / (1 << self.cfg.v_frac)

    @value.setter
    def value(self, v: float) -> None:
        # the engine writes the starting estimate here before a run; the
        # centering accumulators restart with it, as a reset would
        self._st = [value_code(float(v), self.cfg), 0, 0]
        self.v_start = self._st[0]
        self.e_codes, self.r_codes, self.v_codes = [], [], []

    def step(self, err: float, reg: float) -> float:
        e = quantize_err(float(err), self.cfg)
        r = quantize_reg(float(reg), self.cfg)
        v = step_codes(self._st, e, r, self.cfg)
        self.e_codes.append(e)
        self.r_codes.append(r)
        self.v_codes.append(v)
        return self.value

    # ---- kernel view: the ADPLL runs the update inside its per-cycle loop
    def kernel_view(self, n_cycles: int):
        """(step, state, params, record) for the engine's kernel.  The
        state is ``[V, A_reg, A_err]`` as float64 (integers far below
        2^53, so exact); the record holds one (e, r, V) row per cycle."""
        c = self.cfg
        pr = np.array([float(1 << c.e_frac), float(1 << (c.e_bits - 1)),
                       float(c.reg_lsb_hz), float(1 << (c.reg_bits - 1)),
                       float(c.k), float(c.v_max), float(1 << c.v_frac)])
        return (fixed_dp_step, np.array(self._st, dtype=float), pr,
                np.empty((n_cycles, 3)))

    def kernel_done(self, st: np.ndarray, rec: np.ndarray, n_steps: int) -> None:
        """Take the state and the code streams back after a run."""
        self._st = [int(st[0]), int(st[1]), int(st[2])]
        codes = rec[:n_steps].astype(np.int64)
        self.e_codes.extend(codes[:, 0].tolist())
        self.r_codes.extend(codes[:, 1].tolist())
        self.v_codes.extend(codes[:, 2].tolist())
