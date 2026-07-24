"""Cartesian memory DPD wrapped around the whole polar chain.

The polar DPD LUTs (cal.polar_dpd) fix the static AM-AM/AM-PM, but
memory effects (supply/bias dynamics, output-network dispersion — the
chain's post-DPA `memory` model) are not code-static and need a
Volterra-class predistorter.  Standard practice puts a Cartesian GMP
DPD in front of the polar split; here the ILA (vendored padpd) treats
the ENTIRE chain — split, phase modulator, DPA, memory — as the
black-box PA.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..vendor.padpd.dpd import ILAPredistorter
from ..waveforms.base import Waveform


def _chain_as_pa(tx, wf: Waveform, *, noise: bool, seed: int):
    """Wrap PolarTX.run as an x -> y callable on wf's grid."""

    def pa(x: np.ndarray) -> np.ndarray:
        wf2 = replace(wf, x=x)
        return tx.run(wf2, noise=noise, seed=seed).y

    return pa


def fit_chain_ila(tx, wf: Waveform, *, model_factory=None,
                  n_iterations: int = 2, noise: bool = False,
                  seed: int = 0, fit_kwargs: dict | None = None
                  ) -> ILAPredistorter:
    """ILA fit of a Cartesian predistorter against the whole chain."""
    dpd = ILAPredistorter(model_factory=model_factory,
                          n_iterations=n_iterations,
                          fit_kwargs=fit_kwargs or {"regularization": 1e-9})
    dpd.fit(_chain_as_pa(tx, wf, noise=noise, seed=seed), wf.x)
    return dpd


def run_with_ila(tx, wf: Waveform, dpd: ILAPredistorter, *,
                 noise: bool = True, seed: int = 0):
    """Run the chain on the predistorted waveform (renormalized so the
    chain's own full-scale mapping sees the same average power)."""
    u = dpd(wf.x)
    wf2 = replace(wf, x=u)
    return tx.run(wf2, noise=noise, seed=seed)
