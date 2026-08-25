#!/usr/bin/env python3
"""Report what Python an APK actually ships, and optionally gate on it.

Chaquopy packs the Python tree into ``.imy`` archives inside the APK, so
``unzip -l`` on the APK shows you nothing useful.  This walks into them.

Two questions it answers, both of which are otherwise guesswork:

*What is in here?*  Which top-level packages ship, and for each one how many
modules are source (``.py``/``.pyc``) versus native (``.so``).  Useful on an
APK someone hands you, and the fastest honest answer to "is my source
readable in this thing".

*Is this the build I think it is?*  If you produce both an interpreted and a
compiled APK from one workspace, the dangerous failure is the second build
reusing the first's pip output -- the artifacts differ in name only, and no
line of the build log says so.  ``--native`` and ``--pure`` turn the intent
into an assertion that fails the job.

    # report
    python inspect_apk.py app-debug.apk

    # gate: these subpackages must be compiled, these must not be
    python inspect_apk.py compiled.apk --package mylib --native core,solvers
    python inspect_apk.py interpreted.apk --package mylib --pure core,solvers

Exit status is 0 when every requested assertion holds (and, with no
assertions, when every APK contained a readable Python payload).
"""

from __future__ import annotations

import argparse
import io
import zipfile
from collections import defaultdict
from pathlib import Path

SOURCE_SUFFIXES = (".py", ".pyc")


def payload_names(apk: Path) -> list[str]:
    """Every path inside every Chaquopy payload archive in the APK.

    Chaquopy has used ``.imy`` for the app and requirements payloads across
    the versions this was written against; ``.zip`` and ``.mp3`` have both
    appeared historically, so all three are accepted rather than assuming.
    """
    names: list[str] = []
    with zipfile.ZipFile(apk) as z:
        for entry in z.namelist():
            if not entry.endswith((".imy", ".zip", ".mp3")):
                continue
            try:
                with zipfile.ZipFile(io.BytesIO(z.read(entry))) as payload:
                    names += payload.namelist()
            except zipfile.BadZipFile:
                continue          # an .mp3 that is genuinely an .mp3
    return names


def module_key(name: str) -> str:
    """The dotted module a payload entry belongs to, as a path.

    Both halves matter.  The module name is everything up to the first dot,
    so ``solve.cpython-310-x86_64-linux-gnu.so`` and
    ``solve.cpython-310.pyc`` both reduce to ``solve`` -- otherwise a
    compiled module and its bytecode look like unrelated files.  And a
    ``__pycache__`` directory is not a package level; folding it away is what
    keeps ``mylib/core`` counting its own bytecode.
    """
    parent = Path(name).parent
    if parent.name == "__pycache__":
        parent = parent.parent
    stem = Path(name).name.split(".")[0]
    return str(parent / stem) if str(parent) != "." else stem


def tally(names: list[str]) -> dict[str, tuple[int, int]]:
    """path prefix -> (source count, native count), at every level.

    Indexing every directory prefix *and* every module, rather than only the
    top level, is what lets ``--native core/backends`` and ``--native
    core.solve`` both work without the caller knowing the layout.
    """
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for name in names:
        if name.endswith(SOURCE_SUFFIXES):
            slot = 0
        elif name.endswith(".so"):
            slot = 1
        else:
            continue
        key = module_key(name)
        counts[key][slot] += 1
        parts = Path(key).parts[:-1]
        for depth in range(1, len(parts) + 1):
            prefix = "/".join(parts[:depth])
            if prefix != "__pycache__":
                counts[prefix][slot] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


def norm(package: str | None, entry: str) -> str:
    """A --native/--pure entry as a payload path prefix."""
    raw = entry.strip().removesuffix(".py").replace(".", "/")
    return f"{package}/{raw}" if package else raw


def report(apk: Path, counts: dict[str, tuple[int, int]]) -> None:
    tops = sorted(k for k in counts if "/" not in k)
    print(f"{apk.name}:")
    if not tops:
        print("  no Python payload found -- not a Chaquopy APK?")
        return
    for top in tops:
        src, obj = counts[top]
        kind = ("source only" if src and not obj else
                "native only" if obj and not src else
                "mixed" if src and obj else "empty")
        print(f"  {top:<24} {src:>5} source  {obj:>5} native   ({kind})")


def check(counts: dict[str, tuple[int, int]], prefix: str,
          want_native: bool) -> bool:
    """Assert one subtree is compiled (or is not), and explain any failure."""
    if prefix not in counts:
        print(f"  FAIL {prefix}: not present in this APK at all")
        return False
    src, obj = counts[prefix]
    if want_native:
        if obj and not src:
            print(f"  ok   {prefix}: {obj} native, no source")
            return True
        if src and obj:
            print(f"  FAIL {prefix}: {src} source files alongside {obj} "
                  f"native -- source shadows the .so on import, so this "
                  f"build is not compiled in any useful sense")
        else:
            print(f"  FAIL {prefix}: {src} source, 0 native -- this is the "
                  f"interpreted build under a compiled name")
        return False
    if src and not obj:
        print(f"  ok   {prefix}: {src} source, no native")
        return True
    print(f"  FAIL {prefix}: expected source only, found {src} source "
          f"and {obj} native")
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("apks", nargs="+", type=Path)
    ap.add_argument("--package", help="prefix for --native/--pure entries")
    ap.add_argument("--native", default="",
                    help="comma-separated subtrees that must be compiled")
    ap.add_argument("--pure", default="",
                    help="comma-separated subtrees that must NOT be compiled")
    args = ap.parse_args(argv)

    native = [e for e in args.native.split(",") if e.strip()]
    pure = [e for e in args.pure.split(",") if e.strip()]

    ok = True
    for apk in args.apks:
        if not apk.is_file():
            print(f"{apk}: not a file")
            ok = False
            continue
        counts = tally(payload_names(apk))
        report(apk, counts)
        if not counts:
            ok = False
            continue
        for entry in native:
            ok &= check(counts, norm(args.package, entry), True)
        for entry in pure:
            ok &= check(counts, norm(args.package, entry), False)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
