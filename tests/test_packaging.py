"""Packaging claims, checked against the package.

Two facts get stated twice in this repo and can therefore disagree: the
version (pyproject.toml and polartx.__version__) and the promise that the
inline annotations ship (the py.typed marker plus the package-data entry that
actually puts it in the wheel -- either half alone is a silent no-op).

pyproject.toml is parsed with a regex rather than tomllib because tomllib
arrives in 3.11 and this package declares 3.10, which the test-floor CI job
really runs.
"""
from __future__ import annotations

import re
from pathlib import Path

import polartx

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_version_is_declared_once_in_effect():
    """__version__ and the built distribution's version are the same string."""
    m = re.search(r'^version = "([^"]+)"', PYPROJECT, re.M)
    assert m, "pyproject.toml no longer declares a literal version"
    assert polartx.__version__ == m.group(1), (
        f"polartx.__version__ is {polartx.__version__!r} but pyproject.toml "
        f"says {m.group(1)!r}; a release would ship the second one under the "
        "name of the first.")


def test_inline_annotations_are_actually_shipped():
    """PEP 561 needs BOTH the marker file and a package-data entry.

    Dropping either one leaves a package whose annotations are invisible to
    a consumer's type checker, with nothing failing to say so.
    """
    assert (ROOT / "src" / "polartx" / "py.typed").is_file(), (
        "src/polartx/py.typed is missing: without the marker, a consumer's "
        "type checker ignores the annotations CI checks.")
    pkg_data = re.search(r"\[tool\.setuptools\.package-data\](.*?)(\n\[|\Z)",
                         PYPROJECT, re.S)
    assert pkg_data and "py.typed" in pkg_data.group(1), (
        "py.typed exists but no [tool.setuptools.package-data] entry ships "
        "it, so it is absent from the wheel.")
