"""Vendor drift check: the vendored tree matches pinned upstream.

Runs the real ``tools/vendor_check.py`` logic. Skips when the sibling
repos (pll_simulator / PA_DPD) are not checked out — the same graceful
behaviour the CI job relies on — so a bare checkout still passes.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "vendor_check", REPO / "tools" / "vendor_check.py")
vc = importlib.util.module_from_spec(_spec)
sys.modules["vendor_check"] = vc  # so dataclass can resolve annotations
_spec.loader.exec_module(vc)


def _siblings():
    return vc.Siblings({})


def _have_siblings(sib):
    return sib.path("pll_simulator") is not None and sib.path("PA_DPD") is not None


def test_no_drift():
    sib = _siblings()
    if not _have_siblings(sib):
        pytest.skip("sibling repos (pll_simulator/PA_DPD) not present")
    manifest = vc.json.loads(vc.MANIFEST.read_text()) if vc.MANIFEST.exists() else {}
    results = vc.check(sib, manifest)
    drift = [r for r in results if r.status == "DRIFT"]
    assert not drift, "vendor drift:\n" + "\n".join(
        f"  {r.rel}: {r.detail}" for r in drift)


def test_every_file_has_header():
    """Every vendored .py (bar __init__) carries an attribution header —
    this part needs no sibling checkout."""
    bad = []
    for f in sorted((REPO / "src/polartx/vendor").rglob("*.py")):
        if f.name == "__init__.py":
            continue
        first = f.read_text().splitlines()[0]
        if not vc.HEADER.match(first):
            bad.append(str(f.relative_to(REPO)))
    assert not bad, "vendored files missing attribution header: " + ", ".join(bad)


def test_manifest_files_exist():
    """Manifest never references a vendored file that has been removed."""
    if not vc.MANIFEST.exists():
        pytest.skip("no manifest")
    declared = vc.json.loads(vc.MANIFEST.read_text()).get("files", {})
    missing = [rel for rel in declared
               if not (REPO / "src/polartx/vendor" / rel).exists()]
    assert not missing, "stale manifest entries: " + ", ".join(missing)


def test_detects_injected_drift(tmp_path, monkeypatch):
    """A non-import edit to a verbatim file is reported as DRIFT."""
    sib = _siblings()
    if not _have_siblings(sib):
        pytest.skip("sibling repos not present")
    target = REPO / "src/polartx/vendor/pllsim/core/jitter.py"
    original = target.read_text()
    try:
        target.write_text(original + "\nINJECTED_DRIFT = 1\n")
        manifest = vc.json.loads(vc.MANIFEST.read_text())
        results = vc.check(sib, manifest)
        drift = [r for r in results if r.status == "DRIFT"]
        assert any("jitter.py" in r.rel for r in drift)
    finally:
        target.write_text(original)
