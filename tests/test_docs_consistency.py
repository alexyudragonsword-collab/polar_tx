"""The docs that claim to be complete, checked against the package.

`docs/architecture.md` is the module map, and AGENTS.md sends every reader
through it before they touch code.  It had drifted: `cal/`, `metrics/` and
`guiqt/` were summarised one line per DIRECTORY, so 14 module filenames could
not be found there at all.  Nothing was wrong in what it said -- it just
stopped being a map you could look something up in.

A doc that is meant to be exhaustive has to be pinned to the tree, in both
directions: a module with no entry is an omission, and an entry naming a file
that no longer exists is a stale rename.

Counts in prose are the same class of claim.  The one here is the number of
test FUNCTIONS, counted from the AST, and deliberately not the number pytest
prints: collection depends on what optional dependencies are installed
(test_gui_qt.py contributes nothing without PySide6), so the printed total is
a property of the machine, while the function count is a property of the repo.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "polartx"
ARCH = ROOT / "docs" / "architecture.md"
README = ROOT / "README.md"

#: Not part of the module map.  `__init__.py` files are package plumbing (the
#: one in metrics/ that re-exports is described in the map's metrics/ line),
#: and `guiqt/__main__.py` is a two-line `python -m` shim.
EXEMPT = {"__init__.py", "__main__.py"}


def _own_modules() -> list[Path]:
    """Every module polartx owns, vendored copies excluded (they have their
    own section in the map, and their contents are upstream's)."""
    return sorted(p for p in PKG.rglob("*.py")
                  if "vendor" not in p.parts and p.name not in EXEMPT)


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")     # the docs are bilingual


def test_every_module_appears_in_the_architecture_map():
    """A new module without an entry in the map is the drift this catches."""
    doc = _read(ARCH)
    missing = [str(p.relative_to(PKG)) for p in _own_modules()
               if p.name not in doc]
    assert not missing, (
        "docs/architecture.md 2 is the module map and these modules are not "
        f"in it: {missing}.  Add one line each (or add the filename to "
        "EXEMPT here if it is genuinely plumbing).")


def test_the_map_names_no_file_that_does_not_exist():
    """The other direction: a rename that left the map behind."""
    named = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*\.py)", _read(ARCH)))
    real = {p.name for p in PKG.rglob("*.py")} | {
        p.name for p in (ROOT / "tests").glob("*.py")} | {
        p.name for p in (ROOT / "tools").glob("*.py")}
    stale = sorted(named - real)
    assert not stale, (f"docs/architecture.md names files that do not exist: "
                       f"{stale} -- a rename left the map behind.")


def test_readme_test_function_count_is_current():
    """README states how many test functions there are; keep it true.

    Not a snapshot of a physical quantity -- it is a claim about this repo
    that a reader may act on, so it is checked like any other claim.  When it
    fails, update the sentence; the number is supposed to move.
    """
    n = 0
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(_read(p))
        n += sum(1 for x in tree.body
                 if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and x.name.startswith("test_"))
    claimed = re.search(r"# (\d+) 个测试函数", _read(README))
    assert claimed, "README 的 `pytest tests/` 那行不再声明测试函数个数"
    assert int(claimed.group(1)) == n, (
        f"README says {claimed.group(1)} test functions, the tree has {n}. "
        "Update the sentence in README.md 快速开始.")


def test_readme_example_count_matches_the_examples_directory():
    """Same for the example scripts, which the README both counts and CI runs."""
    n = len(list((ROOT / "examples").glob("ex*.py")))
    claimed = re.search(r"(\d+) 个成套脚本", _read(README))
    assert claimed, "README 的 examples 行不再声明脚本个数"
    assert int(claimed.group(1)) == n, (
        f"README says {claimed.group(1)} example scripts, examples/ has {n}.")
