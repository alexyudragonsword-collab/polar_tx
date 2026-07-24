# Vendored from PA_DPD@44f9ee99: src/padpd/pa/hb_import.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Import harmonic-balance / S-parameter simulation results as a PA model.

Pre-tapeout, circuit simulation already yields everything needed for a
first behavioral model:

- an AM-AM / AM-PM table from a harmonic-balance power sweep
  (static nonlinearity),
- S21 tables of the input/output matching networks from S-parameter or
  PSS/PAC analysis (linear memory).

:func:`load_hb_pa` assembles them into a Wiener-Hammerstein model
(FIR -> static LUT nonlinearity -> FIR) — the same structure as the
built-in :class:`ReferencePA` — so the full padpd chain (behavioral
fitting, ILA/DLA DPD, deployment, co-design) can predict PA+DPD system
performance before the PA exists in silicon.

CSV conventions
---------------
AM-AM/AM-PM table (``r_in`` ascending):
    ``r_in,r_out,phase_deg``            (envelope amplitudes, linear)
    or ``pin_dbm,pout_dbm,phase_deg``   (50-ohm powers, converted)
S21 table:
    ``freq_hz,mag_db,phase_deg``        (baseband-relative frequency,
    i.e. offset from the carrier; may cover any sub-range of +/- fs/2,
    values outside the given range hold the edge value)
"""

from __future__ import annotations

import numpy as np

from .base import PAModel


class WienerHammersteinPA(PAModel):
    """FIR -> AM-AM/AM-PM lookup -> FIR, from simulation tables.

    The LUT stores the complex gain ``g(r) = (r_out/r_in) * exp(j*phi)``
    on an ``r_in`` grid; between points it interpolates linearly, above
    the last point it saturates (holds the last ``r_out`` and phase).
    """

    def __init__(self, r_in: np.ndarray, r_out: np.ndarray,
                 phase_deg: np.ndarray,
                 fir_in: np.ndarray | None = None,
                 fir_out: np.ndarray | None = None,
                 drive: float = 1.0):
        self.r_in = np.asarray(r_in, dtype=float)
        self.r_out = np.asarray(r_out, dtype=float)
        self.phase_deg = np.asarray(phase_deg, dtype=float)
        if not np.all(np.diff(self.r_in) > 0):
            raise ValueError("r_in must be strictly ascending")
        self.fir_in = (np.array([1.0 + 0j]) if fir_in is None
                       else np.asarray(fir_in, dtype=complex))
        self.fir_out = (np.array([1.0 + 0j]) if fir_out is None
                        else np.asarray(fir_out, dtype=complex))
        self.drive = float(drive)
        # small-signal gain used to normalize the output level
        self._g0 = self.r_out[0] / max(self.r_in[0], 1e-12)

    def _lut(self, r: np.ndarray) -> np.ndarray:
        """Complex gain at envelope amplitude r (saturating beyond table)."""
        r_c = np.clip(r, self.r_in[0], self.r_in[-1])
        rout = np.interp(r_c, self.r_in, self.r_out)
        # amplitudes above the table saturate: hold the last r_out
        rout = np.where(r > self.r_in[-1], self.r_out[-1], rout)
        ph = np.deg2rad(np.interp(r_c, self.r_in, self.phase_deg))
        safe_r = np.maximum(r, 1e-12)
        return rout / safe_r * np.exp(1j * ph)

    @staticmethod
    def _filt(x: np.ndarray, fir: np.ndarray) -> np.ndarray:
        """Convolve and remove the FIR's center (bulk group) delay.

        S21-derived FIRs are center-tapped; their bulk delay is not part
        of the behavioral model (real captures remove it via alignment),
        so keep only the dispersion around the center tap.
        """
        center = int(np.argmax(np.abs(fir)))
        return np.convolve(x, fir)[center: center + len(x)]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=complex)
        u = self._filt(x, self.fir_in) * self.drive
        v = u * self._lut(np.abs(u))
        w = self._filt(v, self.fir_out)
        return w / (self.drive * self._g0)

    def get_config(self) -> dict:
        return {"r_in": self.r_in.tolist(),
                "r_out": self.r_out.tolist(),
                "phase_deg": self.phase_deg.tolist(),
                "fir_in": [complex(c) for c in self.fir_in],
                "fir_out": [complex(c) for c in self.fir_out],
                "drive": self.drive}


def _read_csv_columns(path: str) -> dict:
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: empty CSV")
    cols = {k.strip().lower(): np.array([float(r[k]) for r in rows])
            for k in rows[0]}
    return cols


def load_amam_table(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read an AM-AM/AM-PM sweep; returns (r_in, r_out, phase_deg)."""
    c = _read_csv_columns(path)
    if "r_in" in c and "r_out" in c:
        r_in, r_out = c["r_in"], c["r_out"]
    elif "pin_dbm" in c and "pout_dbm" in c:
        # 50-ohm peak envelope amplitude: r = sqrt(2 * P * R)
        r_in = np.sqrt(2 * 50 * 10 ** (c["pin_dbm"] / 10) * 1e-3)
        r_out = np.sqrt(2 * 50 * 10 ** (c["pout_dbm"] / 10) * 1e-3)
    else:
        raise ValueError(f"{path}: need r_in/r_out or pin_dbm/pout_dbm")
    phase = c.get("phase_deg", np.zeros_like(r_in))
    order = np.argsort(r_in)
    return r_in[order], r_out[order], phase[order]


def s21_to_fir(path: str, fs: float, n_taps: int = 15) -> np.ndarray:
    """Design an FIR matching a measured/simulated S21 table.

    Frequency sampling: the table (freq_hz relative to the carrier,
    mag_db, phase_deg) is interpolated onto the FFT grid (edge-held
    outside its range), inverse-transformed and windowed to ``n_taps``.
    """
    c = _read_csv_columns(path)
    f_tab, mag_db, ph_deg = c["freq_hz"], c["mag_db"], c["phase_deg"]
    order = np.argsort(f_tab)
    f_tab, mag_db, ph_deg = f_tab[order], mag_db[order], ph_deg[order]

    n = 1024
    f = np.fft.fftfreq(n, d=1 / fs)
    mag = 10 ** (np.interp(f, f_tab, mag_db,
                           left=mag_db[0], right=mag_db[-1]) / 20)
    ph = np.deg2rad(np.interp(f, f_tab, ph_deg,
                              left=ph_deg[0], right=ph_deg[-1]))
    h_t = np.fft.ifft(mag * np.exp(1j * ph))
    m = n_taps
    taps = np.concatenate([h_t[-(m // 2):], h_t[: m - m // 2]])
    taps = taps * np.hanning(m)
    # windowed truncation loses absolute level: recalibrate the DC
    # response to the table's value at f = 0
    target0 = 10 ** (np.interp(0.0, f_tab, mag_db) / 20) * np.exp(
        1j * np.deg2rad(np.interp(0.0, f_tab, ph_deg)))
    return taps * target0 / taps.sum()


def load_hb_pa(amam_csv: str, s21_in_csv: str | None = None,
               s21_out_csv: str | None = None, fs: float = 320e6,
               n_taps: int = 15, drive: float = 1.0
               ) -> WienerHammersteinPA:
    """Assemble a Wiener-Hammerstein PA from HB + S-parameter exports."""
    r_in, r_out, phase = load_amam_table(amam_csv)
    fir_in = s21_to_fir(s21_in_csv, fs, n_taps) if s21_in_csv else None
    fir_out = s21_to_fir(s21_out_csv, fs, n_taps) if s21_out_csv else None
    return WienerHammersteinPA(r_in, r_out, phase,
                               fir_in=fir_in, fir_out=fir_out, drive=drive)
