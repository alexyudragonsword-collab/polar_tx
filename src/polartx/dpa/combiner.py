"""Multi-core / Doherty power combining for the digital PA.

The single-core DPA (``characteristics.efficiency_curve``) carries a
*phenomenological* Doherty efficiency law — a fitted double-hump.  This module
models the combining itself: N sub-PA cores (a main plus one or more peaking
cores) whose currents sum into a shared load, from which BOTH outputs are
*derived* rather than fitted —

* the combined signal — AM-AM / AM-PM including the peaking-handoff kink and
  the distortion from core-to-core **gain / phase imbalance**;
* the drain efficiency — from the two-region **load modulation**, not a curve
  fit: the main core saturates at the backoff point (seeing a load modulated
  up by the impedance inverter) so efficiency rises to η_peak there, then the
  peaking core(s) ramp the output to full power.

Ideal symmetric class-B derivation (main + one peaking, backoff v_t):
  region 1 (x ≤ v_t): main only, load-modulated, P_out = x²,
                      P_dc ∝ x  ->  η = (π/2)·x   -> η_peak = π/4 at x = v_t
  region 2 (x ≥ v_t): main saturated + peaking ramps,
                      P_out and P_dc both linear -> η = π/4 (flat, ideal)
The classic dip between the two efficiency peaks appears only with realistic
class-C peaking (``peaking="C"``), modeled with a ``dip`` term.

Combiner insertion loss scales the output (and efficiency).  Gain/phase
imbalance is the mismatch knob for Monte-Carlo yield.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TWOPI = 2.0 * np.pi


@dataclass
class DohertyCombiner:
    """N-way Doherty / multi-core power combiner.

    ``n_way`` = 2 is the classic main+peaking Doherty; 3 adds a second peaking
    core (extended Doherty) with turn-on points spread over the backoff range.
    ``backoff_db`` is the (first) peaking turn-on, i.e. the low efficiency peak.
    """
    n_way: int = 2
    backoff_db: float = 6.0
    eta_peak: float = 0.785             # class-B ideal = pi/4
    peaking: str = "B"                  # "B" ideal (flat top) | "C" (dip)
    dip: float = 0.12                   # region-2 efficiency dip for class-C
    combiner_loss_db: float = 0.4       # transformer / balun insertion loss
    gain_imbalance: tuple = ()          # per-core fractional gain error (len n_way)
    phase_imbalance_deg: tuple = ()     # per-core phase error [deg] (len n_way)
    seed: int = 0

    def __post_init__(self):
        if self.n_way < 1:
            raise ValueError("n_way >= 1")
        # peaking turn-on points (normalized output amplitude), one per aux core
        self._vt = self._turn_on_points()
        gi = list(self.gain_imbalance) or [0.0] * self.n_way
        pi_ = list(self.phase_imbalance_deg) or [0.0] * self.n_way
        if len(gi) != self.n_way or len(pi_) != self.n_way:
            raise ValueError("gain/phase_imbalance must have length n_way")
        self._gain = np.array(gi, float)
        self._phase = np.deg2rad(np.array(pi_, float))

    def _turn_on_points(self) -> np.ndarray:
        v_bo = 10.0 ** (-self.backoff_db / 20.0)     # first turn-on
        n_aux = self.n_way - 1
        if n_aux <= 0:
            return np.array([])
        # spread aux turn-ons from v_bo up toward 1 (extended Doherty)
        return np.linspace(v_bo, 1.0, n_aux + 1)[:-1]

    # ---------------------------------------------------------- schedule
    def _core_currents(self, x: np.ndarray) -> np.ndarray:
        """Normalized fundamental current of each core vs output amplitude x.

        Returns array (n_way, len(x)).  Core 0 is the main; cores 1.. are
        peaking, each turning on at its point and ramping to full at x=1."""
        x = np.asarray(x, float)
        cores = np.zeros((self.n_way, x.size))
        vt = self._vt
        first = vt[0] if vt.size else 1.0
        # main: ramps 0->1 up to the first turn-on, then saturated at 1
        cores[0] = np.where(x <= first, np.divide(x, first,
                            out=np.zeros_like(x), where=first > 0), 1.0)
        # each peaking core turns on at vt[k] and *saturates at the next
        # turn-on* (or full scale for the last) so the segments sum to x
        bounds = np.concatenate((vt, [1.0]))
        for k in range(vt.size):
            lo, hi = vt[k], bounds[k + 1]
            cores[k + 1] = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        return cores

    def _weights(self) -> np.ndarray:
        """Ideal combiner weights so the (balanced) core sum reproduces x:
        the main spans up to the first turn-on, each peaking core its own
        [turn-on, next-turn-on] segment."""
        vt = self._vt
        if not vt.size:
            return np.array([1.0])
        bounds = np.concatenate((vt, [1.0]))
        return np.concatenate(([vt[0]], np.diff(bounds)))

    # ------------------------------------------------------- signal path
    def combine(self, x: np.ndarray) -> np.ndarray:
        """Complex combined output vs normalized target amplitude x, including
        gain/phase imbalance and insertion loss (ideal balanced -> exactly x)."""
        cores = self._core_currents(x)
        w = self._weights()
        gains = w[:, None] * (1.0 + self._gain[:, None]) * \
            np.exp(1j * self._phase[:, None])
        y = (gains * cores).sum(axis=0)
        return y * 10.0 ** (-self.combiner_loss_db / 20.0)

    def am_curves(self, n: int = 256) -> dict:
        """Sampled AM-AM (normalized) and AM-PM [rad] vs input amplitude."""
        x = np.linspace(0.0, 1.0, n)
        y = self.combine(x)
        amp = np.abs(y)
        amp_n = amp / amp[-1] if amp[-1] > 0 else amp
        ampm = np.unwrap(np.angle(y))
        ampm = ampm - ampm[-1]                      # reference to full scale
        return {"x": x, "amam": amp_n, "ampm_rad": ampm,
                "amam_ripple_db": float(20 * np.log10(
                    np.max(amp_n[1:] / x[1:]) / np.min(amp_n[1:] / x[1:]))),
                "ampm_pp_deg": float(np.rad2deg(np.ptp(ampm)))}

    # -------------------------------------------------------- efficiency
    def efficiency(self, x: np.ndarray) -> np.ndarray:
        """Drain efficiency vs normalized output amplitude, from the two-region
        load-modulation model (see module docstring).  Insertion loss lowers
        the whole curve; class-C peaking adds the inter-peak dip."""
        x = np.asarray(x, float)
        v_t = self._vt[0] if self._vt.size else 1.0
        loss = 10.0 ** (-self.combiner_loss_db / 10.0)     # power loss
        eta = np.where(
            x <= v_t,
            self.eta_peak * np.divide(x, v_t, out=np.zeros_like(x),
                                      where=v_t > 0),        # linear rise
            self.eta_peak)                                   # ideal flat top
        if self.peaking.upper() == "C":
            # realistic class-C peaking: parabolic dip between v_t and 1
            t = np.clip((x - v_t) / (1.0 - v_t), 0.0, 1.0)
            eta = np.where(x > v_t,
                           self.eta_peak * (1.0 - self.dip * 4.0 * t * (1.0 - t)),
                           eta)
        return eta * loss

    def combining_loss_db(self) -> float:
        """Combiner loss at full drive = phase-mismatch loss + insertion loss.

        Referenced to the in-phase (coherent) sum of the *actual* core
        magnitudes, so pure gain imbalance — which is an absorbable gain error,
        not a lost power — contributes nothing; only phase misalignment and the
        transformer insertion loss reduce the delivered power.  Always <= 0."""
        w = self._weights()
        mags = w * (1.0 + self._gain)
        coherent = mags.sum()                          # all cores in phase
        actual = np.abs((mags * np.exp(1j * self._phase)).sum())
        mismatch = 20.0 * np.log10(actual / coherent) if coherent else 0.0
        return float(mismatch - self.combiner_loss_db)

    # ------------------------------------------------- DPA integration
    def to_dpa_specs(self, n: int = 256) -> dict:
        """Return amam / ampm_lut / eff specs to plug into DPAConfig, so a
        multi-core Doherty DPA is a DPA whose characteristics come from the
        combining model:

            c = DohertyCombiner(...); s = c.to_dpa_specs()
            DPAConfig(n_bits=10, amam=s['amam'], ampm_lut=s['ampm_lut'],
                      eff=s['eff'])
        """
        cur = self.am_curves(n)
        xe = np.linspace(0.0, 1.0, n)
        return {
            "amam": ("lut", tuple(cur["x"]), tuple(cur["amam"])),
            "ampm_lut": (tuple(cur["x"]), tuple(np.rad2deg(cur["ampm_rad"]))),
            "eff": ("lut", tuple(xe), tuple(self.efficiency(xe))),
        }


def imbalance_montecarlo(base: DohertyCombiner, *, sigma_gain: float = 0.02,
                         sigma_phase_deg: float = 3.0, n_trials: int = 500,
                         seed: int = 0) -> dict:
    """Monte-Carlo the core-to-core imbalance: distribution of AM-AM ripple,
    AM-PM pp, and full-drive combining loss.  The yield knob for a multi-core
    DPA — how tight the per-core matching must be."""
    rng = np.random.default_rng(seed)
    n = base.n_way
    ripple, ampm_pp, loss = [], [], []
    for _ in range(n_trials):
        c = DohertyCombiner(
            n_way=n, backoff_db=base.backoff_db, eta_peak=base.eta_peak,
            peaking=base.peaking, dip=base.dip,
            combiner_loss_db=base.combiner_loss_db,
            gain_imbalance=tuple(rng.normal(0.0, sigma_gain, n)),
            phase_imbalance_deg=tuple(rng.normal(0.0, sigma_phase_deg, n)))
        cur = c.am_curves()
        ripple.append(cur["amam_ripple_db"])
        ampm_pp.append(cur["ampm_pp_deg"])
        loss.append(c.combining_loss_db())
    # new names rather than rebinding the lists to arrays: same values, and
    # each name keeps one type
    rip_a, ampm_a, loss_a = (np.asarray(v) for v in (ripple, ampm_pp, loss))
    return {
        "amam_ripple_db": {"p50": float(np.percentile(rip_a, 50)),
                           "p95": float(np.percentile(rip_a, 95))},
        "ampm_pp_deg": {"p50": float(np.percentile(ampm_a, 50)),
                        "p95": float(np.percentile(ampm_a, 95))},
        "combining_loss_db": {"p50": float(np.percentile(loss_a, 50)),
                              "p05": float(np.percentile(loss_a, 5))},
        "n_trials": n_trials,
    }


# ====================================================================
# Outphasing (LINC / Chireix) combining
# ====================================================================

@dataclass
class OutphasingCombiner:
    """Two-branch outphasing combiner: isolated (Wilkinson) or Chireix.

    Both branch PAs run at constant, full drive; the output amplitude is set
    purely by the outphasing angle θ between them, ``|y| = 2|s| cos θ``.
    Where the two topologies differ is what happens to the power the load
    does NOT take:

    ``mode="isolated"``
        A Wilkinson / hybrid combiner presents each PA a constant load and
        dumps the difference signal into the isolation resistor.  Delivered
        power ∝ cos²θ while the DC draw is constant, so the drain efficiency
        is ``η_pa · cos²θ`` — the textbook reason isolated outphasing is poor
        at high PAPR.
    ``mode="chireix"``
        A non-isolating combiner lets each PA see the other's current: the
        load admittance seen by branch k is ``Y0 (cos²θ ∓ j sinθ cosθ)``, and a
        compensating shunt reactance ``±jB`` cancels the imaginary part at
        one angle θ_c (``chireix_theta_c_deg``).  A saturated (voltage-source)
        class-B PA into a load of angle ψ runs at ``η_pa · cos ψ``, so
        ``η(θ) = η_pa · cos²θ / sqrt(cos⁴θ + (½ sin 2θ − ½ sin 2θ_c)²)`` — peaks
        of η_pa at θ_c and 90°−θ_c, the classic Chireix double hump.

    Signal path (both modes): the ideal vector sum ``y = g1·y1 + g2·y2``
    scaled by the insertion loss, with ``gain_imbalance`` (fractional) and
    ``phase_imbalance_deg`` per branch — the same three knobs, names and
    units as ``DohertyCombiner``.  The Chireix load modulation is carried
    in the efficiency law only; the amplitude/phase distortion a finite PA
    output impedance would add under load modulation is deliberately not
    modeled at this stage (see ROADMAP).

    Efficiency figures are per-branch-PA peak ``eta_pa`` (the DPA's own
    full-drive number, so the comparison with the polar chain uses the
    same PA model) times the combiner's power factor, times insertion loss.
    """

    mode: str = "isolated"              # "isolated" (Wilkinson) | "chireix"
    chireix_theta_c_deg: float = 60.0   # compensation angle [deg], chireix only
    combiner_loss_db: float = 0.4       # transformer / hybrid insertion loss
    gain_imbalance: tuple = ()          # per-branch fractional gain error (len 2)
    phase_imbalance_deg: tuple = ()     # per-branch phase error [deg] (len 2)

    def __post_init__(self):
        if self.mode not in ("isolated", "chireix"):
            raise ValueError(f"mode must be 'isolated' or 'chireix', got {self.mode!r}")
        gi = list(self.gain_imbalance) or [0.0, 0.0]
        pi_ = list(self.phase_imbalance_deg) or [0.0, 0.0]
        if len(gi) != 2 or len(pi_) != 2:
            raise ValueError("gain/phase_imbalance must have length 2")
        self._gain = np.array(gi, float)
        self._phase = np.deg2rad(np.array(pi_, float))
        # normalized compensating susceptance: cancels the reactive part at θ_c
        self._b = 0.5 * np.sin(2.0 * np.deg2rad(self.chireix_theta_c_deg))

    # ------------------------------------------------------- signal path
    def combine(self, y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
        """Complex combined output of the two constant-envelope branches,
        including gain/phase imbalance and insertion loss.  Balanced and
        lossless this is exactly ``y1 + y2``, i.e. the decomposed signal."""
        g = (1.0 + self._gain) * np.exp(1j * self._phase)
        y = g[0] * np.asarray(y1) + g[1] * np.asarray(y2)
        return y * 10.0 ** (-self.combiner_loss_db / 20.0)

    def combining_loss_db(self) -> float:
        """Loss at θ = 0 (both branches in phase) referenced to the coherent
        sum of the actual branch magnitudes — same convention as
        ``DohertyCombiner``: gain imbalance alone costs nothing, phase
        misalignment and insertion loss do.  Always <= 0."""
        mags = 1.0 + self._gain
        coherent = mags.sum()
        actual = np.abs((mags * np.exp(1j * self._phase)).sum())
        mismatch = 20.0 * np.log10(actual / coherent) if coherent else 0.0
        return float(mismatch - self.combiner_loss_db)

    # -------------------------------------------------------- efficiency
    def power_factor(self, theta: np.ndarray) -> np.ndarray:
        """Efficiency multiplier vs outphasing angle θ [rad], before the
        insertion loss: cos²θ (isolated) or the Chireix load-angle cosine."""
        th = np.asarray(theta, float)
        c2 = np.cos(th) ** 2
        if self.mode == "isolated":
            return c2
        b_eff = 0.5 * np.sin(2.0 * th) - self._b
        return c2 / np.sqrt(c2 * c2 + b_eff * b_eff)

    def efficiency(self, theta: np.ndarray, eta_pa: float) -> np.ndarray:
        """Instantaneous drain efficiency vs θ for branch PAs of peak
        efficiency ``eta_pa``, including insertion loss."""
        loss = 10.0 ** (-self.combiner_loss_db / 10.0)
        return eta_pa * self.power_factor(theta) * loss

    def average_efficiency(self, theta: np.ndarray, eta_pa: float) -> dict:
        """Modulated average over a θ trajectory: ``sum(P_out) / sum(P_dc)``.

        P_out ∝ cos²θ in both modes (the load sees the vector sum).  P_dc
        follows the mode: constant for the isolated combiner (the PAs never
        see the load change), ``sqrt(cos⁴θ + b_eff²) / eta_pa`` for Chireix
        (a voltage-source PA into a load of angle ψ draws current ∝ |Y|).
        Written out per mode rather than as P_out/η so θ → 90° is finite.
        Same keys as ``DPA.average_efficiency`` so the two chains report
        alike; ``backoff_db`` is relative to the two-PA coherent peak.
        """
        th = np.asarray(theta, float)
        c2 = np.cos(th) ** 2
        loss = 10.0 ** (-self.combiner_loss_db / 10.0)
        p_out = c2 * loss
        if self.mode == "isolated":
            p_dc = np.full_like(c2, 1.0 / eta_pa)
        else:
            b_eff = 0.5 * np.sin(2.0 * th) - self._b
            p_dc = np.sqrt(c2 * c2 + b_eff * b_eff) / eta_pa
        tot_dc = float(p_dc.sum())
        return {"eta_avg": float(p_out.sum() / tot_dc) if tot_dc else 0.0,
                "p_out_norm": float(p_out.mean()),
                "backoff_db": float(-10 * np.log10(max(p_out.mean(), 1e-30))),
                "eta_pa": float(eta_pa), "mode": self.mode}
