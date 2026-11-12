"""Compiled (numba) and interpreted kernels give the same records bit for
bit -- with polartx's two extensions to the vendored TDC kernel in use.

The vendored jit compiles every per-cycle kernel when numba is installed
and runs the same functions as plain Python when it is not; upstream's own
tests hold its engines to bit identity between the two.  polartx adds what
upstream does not have: ``mod_freq_dp`` (a clipped direct-point trajectory)
and the ``dp_cal`` step, a kernel passed INTO the TDC kernel as an
argument -- the part numba has to type as a first-class function.  Both are
exercised here, with the float and the fixed-point calibrator.

The backend is chosen at import (``PLLSIM_JIT``), so each side runs in its
own interpreter.  Without numba the test skips; CI's ``test-jit`` job
installs numba and sets ``POLARTX_REQUIRE_NUMBA=1``, which turns that skip
into a failure -- a green run that compared nothing looks exactly like one
that compared something.
"""
import json
import os
import subprocess
import sys

import pytest

_SCRIPT = r"""
import hashlib, json, warnings
import numpy as np
warnings.simplefilter("ignore")
from polartx.vendor.pllsim.core.jit import backend
from polartx.presets import ble_1m_adpll, lte20_adpll
from polartx.vendor.pllsim.calibration.lms import SignSignLMS
from polartx.cal.dp_cal_fixed import FixedSignSignLMS

def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

out = {"backend": backend()}
for name, cal in (("float", SignSignLMS(init=1.05, mu=2e-5)),
                  ("fixed", FixedSignSignLMS(init=1.05))):
    # 200 kHz direct DAC against a 250 kHz GFSK deviation: mod_freq_dp
    # differs from mod_freq on the clipped cycles
    p = ble_1m_adpll(mode="event", dp_gain=1.05, dp_range_hz=200e3)
    p.tx.phasemod.dp_cal = cal
    r = p.tx.run(p.make_waveform(n_bits=200, seed=5), noise=True, seed=3)
    d = r.info["phasemod"]
    out[name] = [h(d["sim"].phase_err_out), h(d["sim"].cal_traces["dp_gain"]),
                 d["dp_clip_frac"] > 0]
    if name == "fixed":
        out["fixed_codes"] = [h(np.asarray(getattr(cal, k), dtype=np.int64))
                              for k in ("e_codes", "r_codes", "v_codes")]
t = lte20_adpll(mode="event")
out["lte"] = h(t.tx.run(t.make_waveform(n_symbols=2, seed=0), noise=True,
                        seed=1).phase_out)
print(json.dumps(out))
"""


def _run(jit: bool) -> dict:
    env = dict(os.environ, PLLSIM_JIT="1" if jit else "0")
    r = subprocess.run([sys.executable, "-c", _SCRIPT], env=env,
                       capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_compiled_and_interpreted_kernels_agree():
    try:
        import numba  # noqa: F401
    except ImportError:
        if os.environ.get("POLARTX_REQUIRE_NUMBA") == "1":
            pytest.fail("POLARTX_REQUIRE_NUMBA=1 but numba is not installed")
        pytest.skip("numba not installed: nothing compiles, nothing to compare")
    compiled, interpreted = _run(True), _run(False)
    assert compiled.pop("backend") == "numba"
    assert interpreted.pop("backend") == "python"
    assert compiled["float"][2] and compiled["fixed"][2]   # the clip was live
    assert compiled == interpreted
