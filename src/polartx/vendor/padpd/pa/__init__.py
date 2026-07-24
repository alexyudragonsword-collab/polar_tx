# Reduced from padpd/pa/__init__.py: registry limited to the vendored models.
import ast

import numpy as np

from .base import PAModel, nmse_db
from .saleh import SalehPA
from .hb_import import (WienerHammersteinPA, load_amam_table, load_hb_pa,
                        s21_to_fir)

_MODEL_CLASSES = {cls.__name__: cls
                  for cls in (SalehPA, WienerHammersteinPA)}


def load_model(path: str) -> PAModel:
    """Load a model saved with :meth:`PAModel.save`."""
    d = np.load(path, allow_pickle=False)
    class_name = str(d["class_name"])
    if class_name not in _MODEL_CLASSES:
        raise ValueError(f"unknown model class in {path}: {class_name}")
    model = _MODEL_CLASSES[class_name](**ast.literal_eval(str(d["config"])))
    if "coeffs" in d:
        model.coeffs = d["coeffs"]
    return model


__all__ = ["PAModel", "load_model", "nmse_db", "SalehPA",
           "WienerHammersteinPA", "load_amam_table", "load_hb_pa",
           "s21_to_fir"]
