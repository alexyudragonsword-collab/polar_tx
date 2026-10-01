# Vendored from PA_DPD@44cbcb3: src/padpd/data/io.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Loaders for external PA data sources.

All loaders return an :class:`IQDataset`. Expected formats are specified
in docs/02_data_interface.md:

- Cadence envelope export: one CSV with columns
  ``time, i_in, q_in, i_out, q_out`` (header names case-insensitive).
- MATLAB capture: a .mat file with complex vectors ``x``, ``y`` and a
  scalar ``fs`` (variable names configurable).
- OpenDPD-style dataset: separate input/output CSVs with ``I,Q`` columns.
"""

from __future__ import annotations

import csv

import numpy as np

from .dataset import IQDataset


def read_csv_columns(path: str) -> dict[str, np.ndarray]:
    """Read a numeric CSV into {lowercased_header: float array}.

    Shared by the Cadence loader, the HB importer and the two-tone table
    loader; utf-8-sig handles BOM'd exports. Raises ValueError on an
    empty or header-only file.
    """
    try:
        cols = _read_csv_columns(path)
    except (StopIteration, IndexError):
        raise ValueError(f"{path}: empty CSV") from None
    if not cols or not len(next(iter(cols.values()))):
        raise ValueError(f"{path}: empty CSV")
    return cols


def _read_csv_columns(path: str) -> dict[str, np.ndarray]:
    # utf-8-sig strips a UTF-8 BOM if present (seen in OpenDPD examples)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        rows = [row for row in reader if row]
    data = np.array(rows, dtype=float)
    return {name: data[:, i] for i, name in enumerate(header)}


def load_cadence_csv(path: str) -> IQDataset:
    """Load a Cadence envelope-simulation CSV export.

    Required columns: time, i_in, q_in, i_out, q_out. The sample rate is
    inferred from the (uniform) time column.
    """
    cols = _read_csv_columns(path)
    required = {"time", "i_in", "q_in", "i_out", "q_out"}
    missing = required - cols.keys()
    if missing:
        raise ValueError(f"Cadence CSV is missing columns: {sorted(missing)}")
    t = cols["time"]
    dt = np.diff(t)
    # atol=0 matters: numpy's default 1e-8 s exceeds the sample period
    # above ~100 MHz, so a jittered export would pass and the sample
    # rate would be taken from its first (arbitrary) step
    if not np.allclose(dt, dt[0], rtol=1e-6, atol=0.0):
        raise ValueError("time column is not uniformly sampled")
    fs = 1.0 / dt[0]
    x = cols["i_in"] + 1j * cols["q_in"]
    y = cols["i_out"] + 1j * cols["q_out"]
    return IQDataset(x, y, fs, {"source": "cadence", "path": path})


def load_matlab_mat(path: str, x_var: str = "x", y_var: str = "y",
                    fs_var: str = "fs") -> IQDataset:
    """Load a MATLAB .mat file with complex vectors and a sample rate."""
    from scipy.io import loadmat
    d = loadmat(path)
    for var in (x_var, y_var, fs_var):
        if var not in d:
            raise ValueError(f".mat file is missing variable '{var}'")
    x = np.asarray(d[x_var]).squeeze()
    y = np.asarray(d[y_var]).squeeze()
    fs = float(np.asarray(d[fs_var]).squeeze())
    return IQDataset(x, y, fs, {"source": "matlab", "path": path})


def load_opendpd_csv(input_path: str, output_path: str,
                     sample_rate_hz: float) -> IQDataset:
    """Load an OpenDPD-style dataset (separate input/output I,Q CSVs)."""
    cin = _read_csv_columns(input_path)
    cout = _read_csv_columns(output_path)
    for cols, path in ((cin, input_path), (cout, output_path)):
        if not {"i", "q"} <= cols.keys():
            raise ValueError(f"{path} must have I,Q columns")
    x = cin["i"] + 1j * cin["q"]
    y = cout["i"] + 1j * cout["q"]
    n = min(len(x), len(y))
    return IQDataset(x[:n], y[:n], sample_rate_hz,
                     {"source": "opendpd",
                      "input_path": input_path, "output_path": output_path})
