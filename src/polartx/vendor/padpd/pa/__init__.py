# Reduced from padpd/pa/__init__.py: registry limited to the vendored models
# (no DDR-Volterra, reference/drifting/thermal PAs, I/Q front-end models).
import ast

import numpy as np

from .base import PAModel, basis_cond, nmse_db
from .gmp import GMPModel
from .hb_import import (WienerHammersteinPA, load_amam_table, load_hb_pa,
                        s21_to_fir)
from .memory_polynomial import MemoryPolynomialModel
from .presets import gmp_opendpd_510, mp_opendpd_500
from .saleh import SalehPA
from .spline import (SplineGMP, SplineMemoryPolynomial, bspline_design_matrix,
                     place_knots)
from .spline_state import CoefficientScheduler, StateConditionedSpline

_MODEL_CLASSES = {cls.__name__: cls
                  for cls in (SalehPA, MemoryPolynomialModel, GMPModel,
                              WienerHammersteinPA, SplineMemoryPolynomial,
                              SplineGMP, StateConditionedSpline)}


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


__all__ = ["PAModel", "load_model", "nmse_db", "basis_cond",
           "mp_opendpd_500", "gmp_opendpd_510",
           "SalehPA", "MemoryPolynomialModel", "GMPModel",
           "WienerHammersteinPA", "SplineMemoryPolynomial", "SplineGMP",
           "StateConditionedSpline", "CoefficientScheduler",
           "bspline_design_matrix", "place_knots",
           "load_amam_table", "load_hb_pa", "s21_to_fir"]
