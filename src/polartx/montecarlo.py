"""Monte Carlo yield analysis over process mismatch and impairment draws.

Each "chip" is a build function seeded with its own RNG stream: draw the
DPA cell mismatch, DTC gain/INL, path skew, direct-path gain error, run
the chain (with whatever calibrations the build enables) and score the
metric.  Serial by default (a WiFi chain run is ~0.1 s; hundreds of
chips are cheap); the per-chip seeds make every run reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class MCResult:
    values: np.ndarray             # metric per chip (e.g. EVM dB)
    limit: float
    seeds: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def yield_frac(self) -> float:
        return float(np.mean(self.values <= self.limit))

    def summary(self) -> dict:
        v = self.values
        return {"n": v.size, "mean": float(v.mean()),
                "std": float(v.std()),
                "p95": float(np.percentile(v, 95)),
                "worst": float(v.max()), "best": float(v.min()),
                "limit": self.limit, "yield": self.yield_frac}


def run_mc(build: Callable[[int], tuple], n_chips: int, *,
           metric: Callable = lambda res: res.evm().db,
           limit: float = -30.0, seed0: int = 1000,
           noise: bool = True) -> MCResult:
    """build(chip_seed) -> (tx, wf); metric(PolarResult) -> float
    (smaller = better, compared against limit)."""
    seeds = seed0 + np.arange(n_chips)
    vals = np.empty(n_chips)
    for i, s in enumerate(seeds):
        tx, wf = build(int(s))
        res = tx.run(wf, noise=noise, seed=int(s) + 7)
        vals[i] = float(metric(res))
    return MCResult(values=vals, limit=limit, seeds=seeds)


def wifi_chip_builder(bw: float = 160e6, qam: int = 1024, *,
                      sigma_cell: float = 0.01,
                      dtc_gain_sigma: float = 0.01,
                      skew_sigma_s: float = 0.5e-9,
                      n_symbols: int = 4,
                      calibrated_skew: bool = False):
    """Reference chip-builder: per-chip DPA mismatch draw, DTC gain
    error draw, AM/PM skew draw; optional ACP-search skew calibration."""
    from .cal.skew import corrected_chain_config, estimate_env_skew
    from .dpa import DPAConfig
    from .presets import wifi_dtc

    wf_cache = {}

    def build(chip_seed: int):
        rng = np.random.default_rng(chip_seed)
        p = wifi_dtc(
            bw=bw, qam=qam,
            dpa=DPAConfig(n_bits=10, n_thermo=6,
                          sigma_cell=sigma_cell, seed=chip_seed),
            env_skew_s=float(rng.normal(0.0, skew_sigma_s)))
        p.tx.phasemod.cfg.gain_error = float(rng.normal(0.0, dtc_gain_sigma))
        if "wf" not in wf_cache:
            wf_cache["wf"] = p.make_waveform(n_symbols=n_symbols, seed=0)
        wf = wf_cache["wf"]
        if calibrated_skew:
            res = p.tx.run(wf, noise=False)
            est = estimate_env_skew(res)
            p.tx.cfg = corrected_chain_config(p.tx.cfg, est["skew_s"])
        return p.tx, wf

    return build
