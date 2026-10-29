"""Smoke-test a polartx desktop host: the frozen exe, or the package itself.

    python packaging/smoke_webhost.py dist/polartx-gui.exe
    python packaging/smoke_webhost.py python -m polartx.webhost   # same checks

Starts the command with ``POLARTX_NO_BROWSER=1``, waits for it to serve, and
then asks it what a user's first minutes would: the page and every file it
loads, the bridge's boot call, one workbench run, and the desktop-only RTL
export.  "Served HTTP 200" alone was the old web exe's check; a frozen exe
that serves its index but lost a data file, a lazily imported module or
matplotlib's data still passes that and fails on the first click.

Exit status 0 only if every check passed.  Standard library only, so it runs
on a bare CI runner before anything is installed.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

PORT = 8765
URL = f"http://127.0.0.1:{PORT}/"
# loopback only; a proxy in the environment must not carry these away
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get(path: str, timeout: float = 10) -> bytes:
    with _OPENER.open(URL + path, timeout=timeout) as r:
        return r.read()


def _call(method: str, args: dict | None = None, timeout: float = 600) -> dict:
    req = urllib.request.Request(
        URL + "call", headers={"Content-Type": "application/json"},
        data=json.dumps({"method": method, "args": args or {}}).encode())
    with _OPENER.open(req, timeout=timeout) as r:
        env = json.loads(r.read())
    if not env.get("ok"):
        raise RuntimeError(f"{method}: {env.get('error')}\n{env.get('traceback', '')}")
    return env["result"]


def _checks() -> None:
    index = _get("").decode("utf-8")
    assets = re.findall(r'(?:src|href)="([^":]+\.(?:js|css|svg|png|ico))"', index)
    if not assets:
        raise RuntimeError("index.html references no scripts or styles")
    for a in assets:
        _get(a.lstrip("./"))
    print(f"page: index + {len(assets)} assets ({', '.join(assets)})")

    v = _call("version", timeout=300)
    print(f"bridge: polartx {v['lib']}, Python {v['python']}, numpy {v['numpy']}")

    p = _call("list_presets")
    r = _call("chain", {"preset": p["default"], "png": False})
    evm = r["metrics"].get("EVM [dB]")
    print(f"workbench: {p['default']} -> {len(r['metrics'])} metrics, "
          f"EVM {evm} dB, {len(r.get('plots', []))} plot panels")
    # a number, not a range: what EVM a preset reaches is the test suite's
    # business; here a missing or non-finite one means the frozen run broke
    if not isinstance(evm, (int, float)) or evm != evm or not r.get("plots"):
        raise RuntimeError("the workbench run came back without EVM or plots")

    with tempfile.TemporaryDirectory() as out:
        rtl = _call("rtl_export", {"outdir": out, "verify": False})
    print(f"rtl_export: {len(rtl['files'])} files")
    if not rtl["files"]:
        raise RuntimeError("the RTL export wrote nothing")


def main(cmd: list[str], wait_s: float = 300) -> int:
    if os.path.isfile(cmd[0]):
        # CreateProcess does not search a relative path with "/" in it
        cmd = [os.path.abspath(cmd[0]), *cmd[1:]]
    env = dict(os.environ, POLARTX_NO_BROWSER="1")
    log = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    ok = False
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < wait_s:
            if proc.poll() is not None:
                print(f"the host exited with {proc.returncode} before serving")
                break
            try:
                _get("", timeout=5)
            except OSError:
                time.sleep(2)
                continue
            print(f"serving after {time.monotonic() - t0:.0f} s")
            _checks()
            ok = proc.poll() is None
            if not ok:
                print("the host died during the checks")
            break
        else:
            print(f"nothing served on {URL} within {wait_s:.0f} s")
    except Exception as exc:                     # noqa: BLE001 (reported)
        print(f"FAILED: {exc}")
    finally:
        proc.kill()
        proc.wait()
        log.seek(0)
        print("== host output ==")
        print(log.read().decode("utf-8", "replace"))
    print("SMOKE OK" if ok else "SMOKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1:]))
