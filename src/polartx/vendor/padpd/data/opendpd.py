# Vendored from PA_DPD@44f9ee99: src/padpd/data/opendpd.py
# Adapted-copy policy: see src/polartx/vendor/__init__.py
"""Loader for OpenDPD-format dataset folders.

An OpenDPD dataset folder (https://github.com/lab-emi/OpenDPD) contains a
``spec.json`` with measurement metadata plus the I/Q data in one of two
layouts:

- ``split_csv``: six files ``{train,val,test}_{input,output}.csv``, each
  with columns ``I,Q`` (pre-split, contiguous, time-aligned).
- ``single_csv``: one file (default ``data.csv``) with columns
  ``I_in,Q_in,I_out,Q_out``; contiguous train/val/test split by the
  ``split_ratios`` in the spec (default 0.6/0.2/0.2).

Key spec fields: ``input_signal_fs`` (sample rate), ``bw_main_ch``,
``bw_sub_ch``, ``n_sub_ch``, ``nperseg`` (segment length used by OpenDPD
metrics), ``modulation``. The full spec is attached to each returned
dataset's ``meta["spec"]`` so OpenDPD-compatible metrics can be computed
without re-reading the folder.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .dataset import IQDataset
from .io import _read_csv_columns

SPLITS = ("train", "val", "test")


def _spec_of(dataset_dir: str) -> dict:
    spec_path = os.path.join(dataset_dir, "spec.json")
    if not os.path.exists(spec_path):
        return {}
    with open(spec_path, encoding="utf-8-sig") as f:
        return json.load(f)


def _read_iq(path: str) -> np.ndarray:
    cols = _read_csv_columns(path)
    if not {"i", "q"} <= cols.keys():
        raise ValueError(f"{path} must have I,Q columns")
    return cols["i"] + 1j * cols["q"]


def load_opendpd_dataset(dataset_dir: str,
                         sample_rate_hz: float | None = None) -> dict:
    """Load an OpenDPD dataset folder.

    Returns ``{"train": IQDataset, "val": IQDataset, "test": IQDataset,
    "spec": dict}``. ``sample_rate_hz`` overrides/replaces the spec's
    ``input_signal_fs`` (required if the folder has no spec.json).
    """
    spec = _spec_of(dataset_dir)
    fs = sample_rate_hz or spec.get("input_signal_fs")
    if fs is None:
        raise ValueError(
            f"{dataset_dir} has no spec.json with input_signal_fs; "
            "pass sample_rate_hz explicitly")

    fmt = spec.get("dataset_format", "split_csv")
    if fmt == "split_csv":
        parts = {}
        for split in SPLITS:
            x = _read_iq(os.path.join(dataset_dir, f"{split}_input.csv"))
            y = _read_iq(os.path.join(dataset_dir, f"{split}_output.csv"))
            parts[split] = (x, y)
    elif fmt == "single_csv":
        csv_path = os.path.join(dataset_dir,
                                spec.get("csv_filename", "data.csv"))
        cols = _read_csv_columns(csv_path)
        required = {"i_in", "q_in", "i_out", "q_out"}
        if not required <= cols.keys():
            raise ValueError(f"{csv_path} must have columns I_in,Q_in,"
                             f"I_out,Q_out")
        x = cols["i_in"] + 1j * cols["q_in"]
        y = cols["i_out"] + 1j * cols["q_out"]
        ratios = spec.get("split_ratios", {"train": 0.6, "val": 0.2})
        n = len(x)
        n_train = int(n * ratios.get("train", 0.6))
        n_val = int(n * ratios.get("val", 0.2))
        parts = {
            "train": (x[:n_train], y[:n_train]),
            "val": (x[n_train:n_train + n_val], y[n_train:n_train + n_val]),
            "test": (x[n_train + n_val:], y[n_train + n_val:]),
        }
    else:
        raise ValueError(f"unknown dataset_format: {fmt}")

    name = os.path.basename(os.path.normpath(dataset_dir))
    result = {"spec": spec}
    for split, (x, y) in parts.items():
        result[split] = IQDataset(x, y, float(fs), meta={
            "source": "opendpd", "dataset": name, "split": split,
            "spec": spec,
        })
    return result
