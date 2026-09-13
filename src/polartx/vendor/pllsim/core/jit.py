# Vendored from pll_simulator@931cfaf: src/pllsim/core/jit.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""The one decorator every per-cycle kernel wears, with or without numba.

The six time-domain engines run one Python iteration per reference cycle,
and for years that iteration was a chain of method calls on block objects:
4-40 us a cycle, 1-6 s for the 40k-150k cycles the GUIs default to, times
the run count of a Monte Carlo or a hop sweep.  Every engine's loop and
every block's per-cycle arithmetic is now a plain function of floats and
arrays -- a *kernel* -- and this module decides how it runs:

* ``pip install pllsim[fast]`` brings numba, and every kernel is compiled to
  machine code on first use (and cached next to the module, so the second
  process pays nothing).  Measured 2026-09: 30-100x on the loop, which
  leaves the FFT post-processing as the largest remaining cost.
* Without numba the very same functions run as ordinary Python.  There is
  no second implementation to keep in step: ``pure(f)`` hands back the
  un-compiled twin of a compiled kernel, and ``tests/test_kernels.py`` runs
  every preset both ways and requires the records to be bit-identical.
* ``PLLSIM_JIT=0`` in the environment forces the Python path even when
  numba is installed -- the switch the bit-identity test uses, and the one
  to reach for when a traceback inside a kernel is wanted.

Rules a kernel has to follow for the two paths to stay bit-identical
(each one was found by breaking it):

* Scalar maths goes through ``math.*``/``cmath.*``, never ``np.*`` on a
  scalar and never ``**``: numba lowers ``np.exp`` and ``x ** 2`` to its
  own implementations, which differ from CPython's in the last bit (849 and
  20 of 20 000 draws respectively); ``math.exp`` and ``x * x`` do not.
* No numpy array operation inside the per-cycle path.  ``a @ x`` goes to
  BLAS, whose fused multiply-adds differ from an explicit loop in ~half of
  all 2x2 products; explicit loops agree between numba and Python in all
  of 20 000 trials.  Small state vectors are updated element by element.
* The interpreted path reads numpy scalars out of the state arrays, and
  those follow numpy's arithmetic, not CPython's.  Float ops are the same
  IEEE ops either way; complex *division* is not (numpy multiplies by a
  reciprocal), so it is spelled out in real arithmetic (loopfilter.cdiv).
* Random numbers come from a standard-normal pool drawn *before* the loop
  and consumed through a cursor, in the order the block objects used to
  draw them.  numpy's Generator gives the same stream whether asked one
  sample at a time or all at once, so a run's noise did not change when the
  loops moved into kernels.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])


def _wanted() -> bool:
    return os.environ.get("PLLSIM_JIT", "1").strip().lower() not in (
        "0", "off", "no", "false", "")


def _load_njit() -> Callable[..., Any] | None:
    """numba's ``njit``, or None when it is absent or switched off.

    Written as a function that returns rather than a module-level import
    with a None fallback: rebinding the imported name is an assignment
    mypy reads as "None into an overloaded function", and it only sees it
    where numba's own stubs are installed -- which is CI, not necessarily
    the machine the change was written on.
    """
    if not _wanted():
        return None
    try:
        from numba import njit
    except ImportError:
        return None
    return njit


_njit = _load_njit()
BACKEND = "numba" if _njit is not None else "python"


def kernel(fn: _F) -> _F:
    """Compile ``fn`` with numba when it is available; otherwise return it as is.

    Compiled lazily on first call, cached on disk next to the module.  A
    kernel is a module-level function of scalars and arrays only.
    """
    if _njit is None:
        return fn
    return _njit(cache=True)(fn)


def backend() -> str:
    """``"numba"`` when kernels compile, ``"python"`` when they interpret."""
    return BACKEND


def pure(fn: Callable[..., Any]) -> Callable[..., Any]:
    """The un-compiled twin of a kernel (the function itself on the Python path)."""
    return getattr(fn, "py_func", fn)


def sgn(x: float) -> float:
    """``np.sign`` for a float, spelled so both paths agree bit for bit."""
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


sgn = kernel(sgn)
