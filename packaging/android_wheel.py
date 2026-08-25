#!/usr/bin/env python3
"""Build a Chaquopy-installable wheel with part of a Python package compiled to
native code.

Why this exists
---------------
An APK is a zip, Chaquopy's payload inside it is a zip, and ``.pyc`` gives back
function names, line numbers and whole docstrings to anyone who runs
``strings``.  Cythonising the modules you care about and shipping ``.so``
replaces that with machine code.

Be clear about what it does not buy before you spend a day on it: anything you
leave uncompiled still ships as bytecode, WebView assets stay plain text, and
literal constants survive compilation in the constant pool where a short script
finds them exactly.  This raises the cost of reading your *algorithms*.  It is
not a licence check and it is not obfuscation of your data.

How it works
------------
Every input is public and standard, which is why this is a few hundred lines
rather than a fork of Chaquopy:

* Android CPython -- headers and ``libpythonX.Y.so`` -- is published on **Maven
  Central** as ``com.chaquo.python:target``, not behind chaquo.com.  So the
  wheel build works even where the Gradle side's package index does not.
* The compiler is the plain NDK clang wrapper.  Chaquopy's own
  ``target/android-env.sh`` does nothing more exotic.
* The wheel tag format comes from Chaquopy's ``server/pypi/build-wheel.py``:
  ``cp310-cp310-android_21_arm64_v8a``.
* Cythonised pure-Python modules need **only** ``Python.h``.  Third-party
  binary dependencies (numpy, scipy, …) stay as Chaquopy's own prebuilt
  wheels; nothing here rebuilds them.

One version caveat: on CPython 3.10 the generated C stays on the public API,
because Cython's ``internal/pycore_frame.h`` include sits behind
``PY_VERSION_HEX >= 0x030b00a6``.  On 3.11+ that include is live, so the
headers must be the exact target build rather than merely the right minor
version.

Usage
-----
    # cross-compile for a device, from your project root
    python android_wheel.py --package mylib --compile core,solvers \
        --abi arm64-v8a --ndk "$ANDROID_NDK_HOME"

    # same logic, this machine's compiler, no NDK -- how to test changes here
    python android_wheel.py --package mylib --compile core,solvers --host

``--package`` is auto-detected when the project has exactly one importable
package.  ``--compile`` is deliberately **not** optional and has no "everything"
default: the set you compile is a claim that your test suite still passes with
those ``.py`` files deleted, and a default would make that claim for you.  Pass
``--compile all`` if you really mean the whole tree, then go prove it.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import shutil
import subprocess
import sys
import sysconfig
import urllib.request
import zipfile
from pathlib import Path

#: Chaquopy resolves a two-part ``version = "3.10"`` to a full target build.
#: Read the mapping from Chaquopy's ``Common.java`` at the plugin tag you use;
#: this default is what 15.0.1 resolves "3.10" to.
DEFAULT_TARGET = "3.10.13-0"

#: The NDK sysroot API level to compile against.  Distinct from the app's
#: ``minSdk`` -- Chaquopy's own floor is 21 and its wheel tags say so, so
#: matching it keeps the tag honest even when your app requires more.
DEFAULT_API_LEVEL = 21

MAVEN = ("https://repo.maven.apache.org/maven2/com/chaquo/python/target/"
         "{v}/target-{v}-{abi}.zip")

#: ABI -> the clang target triplet the NDK names its wrappers with.
TRIPLETS = {
    "arm64-v8a": "aarch64-linux-android",
    "x86_64": "x86_64-linux-android",
    "armeabi-v7a": "armv7a-linux-androideabi",
    "x86": "i686-linux-android",
}


def run(cmd: list[str], **kw) -> None:
    print("  $", " ".join(str(c) for c in cmd[:6]),
          "…" if len(cmd) > 6 else "", flush=True)
    subprocess.run(cmd, check=True, **kw)


# --------------------------------------------------------------- discovery

def find_package(project: Path, name: str | None) -> Path:
    """Locate the importable package directory to compile.

    Handles both ``src/`` and flat layouts.  When the name is not given this
    insists on there being exactly one candidate rather than picking: guessing
    wrong here produces a wheel that installs and imports nothing.
    """
    roots = [project / "src", project]
    if name:
        for root in roots:
            cand = root / name
            if (cand / "__init__.py").is_file():
                return cand
        raise SystemExit(f"no package {name!r} under {project}/src or {project}")
    found = []
    for root in roots:
        if not root.is_dir():
            continue
        for cand in sorted(root.iterdir()):
            if (cand / "__init__.py").is_file() and not cand.name.startswith("."):
                found.append(cand)
        if found:
            break
    if len(found) != 1:
        names = ", ".join(p.name for p in found) or "none"
        raise SystemExit(f"--package is required: candidates are {names}")
    return found[0]


def select_sources(tree: Path, spec: list[str]) -> list[Path]:
    """The ``.py`` files to compile, from a list of subpackage names.

    ``all`` means the whole tree.  Anything else is a path relative to the
    package, so ``core`` and ``core/backends`` both work; a name that matches
    nothing is an error rather than a silent no-op, because "I compiled it"
    and "I typo'd it" otherwise look identical in the output.
    """
    if spec == ["all"]:
        sources = sorted(tree.rglob("*.py"))
    else:
        sources = []
        for entry in spec:
            # entries may be dotted (core.backends) or slashed
            # (core/backends), with an optional .py -- so strip the suffix
            # before turning dots into separators, or "ui/form.py" becomes
            # the directory "ui/form/py" and matches nothing
            raw = entry.strip()
            if raw.endswith(".py"):
                raw = raw[: -len(".py")]
            target = tree / raw.replace(".", "/")
            if target.is_dir():
                hits = sorted(target.rglob("*.py"))
            elif target.with_suffix(".py").is_file():
                hits = [target.with_suffix(".py")]
            else:
                raise SystemExit(f"--compile {entry!r} matches nothing "
                                 f"under {tree}")
            if not hits:
                raise SystemExit(f"--compile {entry!r} contains no .py files")
            sources += hits
    if not sources:
        raise SystemExit(f"nothing to compile under {tree}")
    return sorted(set(sources))


# ------------------------------------------------------------- the toolchain

def fetch_target(abi: str, target_version: str, work: Path) -> tuple[Path, Path]:
    """Android CPython headers and libpython, straight from Maven Central."""
    url = MAVEN.format(v=target_version, abi=abi)
    zip_path = work / f"target-{abi}.zip"
    if not zip_path.exists():
        print(f"  fetching {url}", flush=True)
        urllib.request.urlretrieve(url, zip_path)
    dest = work / f"target-{abi}"
    if not dest.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
    xy = ".".join(target_version.split(".")[:2])
    include = dest / "include" / f"python{xy}"
    libdir = dest / "jniLibs" / abi
    if not include.is_dir():
        raise SystemExit(f"no python{xy} headers in {dest}; "
                         f"is --target-version right?")
    if not libdir.is_dir():
        raise SystemExit(f"no jniLibs/{abi} in {dest}; wrong ABI name?")
    return include, libdir


def cythonize(package_name: str, tree: Path, sources: list[Path]) -> list[Path]:
    """Translate the chosen modules to C, in place.

    ``--no-docstrings`` is not cosmetic.  Cython keeps docstrings by default,
    and they are the single most readable thing left in a compiled module --
    without this flag ``strings`` on the result prints your module docstrings
    back verbatim.  Note it is a real flag; ``-X docstrings=False`` is not, and
    fails in a way that leaves a working build that protects nothing.
    """
    for py in sources:
        # the module name must be the full dotted path, or the generated init
        # function is named for the bare file and the import fails at runtime
        rel = py.relative_to(tree).with_suffix("")
        modname = ".".join((package_name, *rel.parts))
        run([sys.executable, "-m", "cython", "-3", "--no-docstrings",
             "--module-name", modname,
             "-o", str(py.with_suffix(".c")), str(py)])
    return [p.with_suffix(".c") for p in sources]


def compile_c(csrc: Path, include_dirs: list[Path], cc: str,
              extra: list[str], libdir: Path | None, pylib: str) -> Path:
    """Compile one cythonised module to a shared object.

    Three flags below are load-bearing and were each added after a real
    failure or to convert one into a visible failure:

    ``-include complex.h``
        Cython lowers ``x ** y`` on values it cannot prove non-negative to
        ``cpow()`` (see ``__Pyx_c_pow_double``) without including
        ``<complex.h>``.  glibc declares those functions through headers
        Python.h already drags in, so a host build never notices; the NDK's
        clang does not, and treats an implicit declaration as an ERROR
        rather than a warning under C99+.  In this project 13 of the 42
        compiled modules generate that call, so it is not one bad module to
        exclude — it is the toolchain that has to be told.

    ``-lm``
        Android keeps the math library separate from libc, and the flag
        below means undefined math symbols can no longer be deferred.

    ``-Wl,--no-undefined``, only when linking libpython
        A ``-shared`` link happily leaves symbols undefined, so a missing
        platform function becomes a dlopen failure on a phone — the one
        class of breakage a clean cross-compile is famous for hiding.  With
        this, that becomes a link error in CI instead.

        It applies ONLY on the cross path, where ``-lpython`` is passed:
        a host build deliberately does not link libpython (a CPython
        extension module resolves the C-API from the interpreter that loads
        it), so requiring every symbol there would fail on the Python API
        itself.  Chaquopy ships libpython as a real shared library and the
        wheel links against it, which is what makes the check meaningful
        there and wrong here.
    """
    so = csrc.with_suffix(".so")
    cmd = [cc, "-shared", "-fPIC", "-O2", "-fvisibility=hidden",
           "-include", "complex.h"]
    for inc in include_dirs:
        cmd += ["-I", str(inc)]
    cmd += extra + ["-o", str(so), str(csrc)]
    if libdir is not None:
        cmd += ["-L", str(libdir), f"-l{pylib}", "-Wl,--no-undefined"]
    cmd += ["-lm"]
    run(cmd)
    return so


# ----------------------------------------------------------------- the wheel

def build_tree(src: Path, work: Path) -> Path:
    """A copy of the package to mutate, so the real source tree is untouched."""
    tree = work / src.name
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(src, tree, ignore=shutil.ignore_patterns("__pycache__"))
    return tree


def wheel_metadata(project: Path, work: Path) -> Path:
    """Reuse the project's real dist-info rather than hand-writing METADATA.

    Dependencies, requires-python, the description and the licence all live in
    pyproject.toml already.  Regenerating them here would be a second copy,
    and second copies drift.
    """
    out = work / "refwheel"
    if not out.exists():
        out.mkdir(parents=True)
        run([sys.executable, "-m", "build", "--wheel",
             "--outdir", str(out), str(project)])
    wheels = sorted(out.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one reference wheel in {out}, "
                         f"found {len(wheels)}")
    extracted = work / "refwheel-x"
    if not extracted.exists():
        with zipfile.ZipFile(wheels[0]) as z:
            z.extractall(extracted)
    infos = sorted(extracted.glob("*.dist-info"))
    if len(infos) != 1:
        raise SystemExit(f"expected one .dist-info in {extracted}")
    return infos[0]


def assemble(package_name: str, tree: Path, dist_info: Path,
             tag: str, outdir: Path) -> Path:
    """Zip the package plus a retagged dist-info into a wheel.

    Hand-assembled rather than driven through setuptools: the tree already
    contains exactly what should ship, and ``package-data`` would have to be
    taught about ``.so`` files that only exist during this build.
    """
    # strip ".dist-info" before splitting: splitting the raw directory name on
    # "-" makes the version "1.2.3.dist", and pip rejects a wheel whose
    # filename does not parse
    stem = dist_info.name[: -len(".dist-info")]
    dist_name, version = stem.split("-", 1)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{dist_name}-{version}-{tag}.whl"
    records: list[tuple[str, str, int]] = []

    def add(z: zipfile.ZipFile, arc: str, data: bytes) -> None:
        z.writestr(arc, data)
        digest = base64.urlsafe_b64encode(
            hashlib.sha256(data).digest()).rstrip(b"=").decode()
        records.append((arc, f"sha256={digest}", len(data)))

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(tree.rglob("*")):
            if path.is_dir() or path.suffix in {".c", ".pyc"}:
                continue
            add(z, str(Path(package_name) / path.relative_to(tree)),
                path.read_bytes())
        # rglob, not iterdir: modern setuptools puts the licence under a
        # `licenses/` subdirectory, and treating that directory as a file
        # aborts the build
        for path in sorted(dist_info.rglob("*")):
            if path.is_dir() or path.name == "RECORD":
                continue
            data = path.read_bytes()
            if path.name == "WHEEL":
                # the tag is the whole point: pip picks a wheel by it, and a
                # py3-none-any tag would let it install on the wrong ABI
                lines = [ln for ln in data.decode().splitlines()
                         if not ln.startswith(("Tag:", "Root-Is-Purelib:"))]
                lines += ["Root-Is-Purelib: false", f"Tag: {tag}"]
                data = ("\n".join(lines) + "\n").encode()
            add(z, str(Path(dist_info.name) / path.relative_to(dist_info)), data)
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        for row in records:
            w.writerow(row)
        w.writerow([f"{dist_info.name}/RECORD", "", ""])
        z.writestr(f"{dist_info.name}/RECORD", buf.getvalue())
    return out


# ------------------------------------------------------------------- driver

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=".",
                    help="project root holding pyproject.toml (default: .)")
    ap.add_argument("--package",
                    help="importable package name; auto-detected when the "
                         "project has exactly one")
    ap.add_argument("--compile", dest="compile_spec", required=True,
                    help="comma-separated subpackages or modules to compile, "
                         "relative to the package; or 'all'")
    ap.add_argument("--abi", choices=sorted(TRIPLETS))
    ap.add_argument("--ndk", default=os.environ.get("ANDROID_NDK_HOME", ""))
    ap.add_argument("--host", action="store_true",
                    help="compile for this machine instead: exercises every "
                         "step but the cross-compiler, with no NDK")
    ap.add_argument("--target-version", default=DEFAULT_TARGET,
                    help=f"Chaquopy CPython target build "
                         f"(default: {DEFAULT_TARGET})")
    ap.add_argument("--api-level", type=int, default=DEFAULT_API_LEVEL)
    ap.add_argument("--work", help="scratch dir (default: <project>/build-android-wheel)")
    ap.add_argument("--outdir", help="where the wheel lands "
                                     "(default: <project>/android/app/pysrc)")
    args = ap.parse_args()
    if not args.host and not args.abi:
        ap.error("--abi is required unless --host is given")

    project = Path(args.project).resolve()
    work = Path(args.work) if args.work else project / "build-android-wheel"
    outdir = Path(args.outdir) if args.outdir else project / "android/app/pysrc"
    work.mkdir(parents=True, exist_ok=True)

    src = find_package(project, args.package)
    package_name = src.name
    print(f"package {package_name} at {src}")

    tree = build_tree(src, work)
    sources = select_sources(tree, args.compile_spec.split(","))
    csrcs = cythonize(package_name, tree, sources)
    print(f"cythonised {len(csrcs)} modules", flush=True)

    xy = ".".join(args.target_version.split(".")[:2])
    py_tag = "cp" + xy.replace(".", "")
    if args.host:
        include_dirs = [Path(sysconfig.get_paths()["include"])]
        cc, extra, libdir = os.environ.get("CC", "cc"), [], None
        plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        # the *running* interpreter's tag, not the Android one: using the
        # target tag here produces e.g. a cp310 wheel on a 3.11 host, which
        # pip then refuses as unsupported
        host_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        tag = f"{host_tag}-{host_tag}-{plat}"
    else:
        if not args.ndk or not Path(args.ndk).is_dir():
            raise SystemExit("--ndk must point at an NDK "
                             "(or set ANDROID_NDK_HOME)")
        include, libdir = fetch_target(args.abi, args.target_version, work)
        include_dirs = [include]
        toolchain = next((Path(args.ndk) / "toolchains/llvm/prebuilt").iterdir())
        cc = str(toolchain / "bin"
                 / f"{TRIPLETS[args.abi]}{args.api_level}-clang")
        if not Path(cc).exists():
            raise SystemExit(f"no compiler at {cc}")
        extra = []
        tag = (f"{py_tag}-{py_tag}-android_{args.api_level}_"
               f"{args.abi.replace('-', '_')}")

    for csrc in csrcs:
        compile_c(csrc, include_dirs, cc, extra, libdir, f"python{xy}")
    # the .py must go, or Python imports it in preference on some paths and
    # the whole exercise silently does nothing
    for csrc in csrcs:
        csrc.with_suffix(".py").unlink()
        csrc.unlink()

    whl = assemble(package_name, tree, wheel_metadata(project, work),
                   tag, outdir)
    print(f"\n{whl}  ({whl.stat().st_size / 1024:.0f} KiB, tag {tag})")

    # Verify the artifact, do not assume it.  A .py surviving beside its .so
    # shadows it on import, and the build log looks identical either way.
    compiled_rel = {str(p.relative_to(tree).with_suffix("")) for p in sources}
    with zipfile.ZipFile(whl) as z:
        names = z.namelist()
    sos = [n for n in names if n.endswith(".so")]
    strays = [n for n in names if n.endswith(".py")
              and str(Path(n).relative_to(package_name).with_suffix(""))
              in compiled_rel]
    print(f"  {len(sos)} compiled modules, {len(strays)} .py left behind")
    if strays:
        raise SystemExit(f"a .py survived a compiled module ({strays[0]}) -- "
                         f"it would shadow the .so and protect nothing")
    if len(sos) != len(sources):
        raise SystemExit(f"expected {len(sources)} .so, found {len(sos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
