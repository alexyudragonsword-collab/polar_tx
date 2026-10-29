"""The desktop and browser host for the polartx page.

The phone shows ``webui/www`` in a WebView and reaches Python through a
JavaScript interface; this module gives a laptop the same page and the same
bridge over a local HTTP server, so all three hosts run one front end:

* ``GET /`` and ``GET /<file>`` serve ``webui/www`` (the files the APK
  bundles as its assets);
* ``POST /call`` with ``{"method": ..., "args": {...}}`` answers with
  exactly what :func:`polartx.appbridge.call` returns — the same envelope
  the phone receives — plus the few :data:`DESKTOP_METHODS` that only make
  sense on a computer (the RTL export).

Standard library only, bound to 127.0.0.1, and calls are serialized: the
page already coalesces its own requests, and matplotlib's pyplot (which a
few report methods still use for their PNG) is not thread-safe.

The desktop gets the page in a Chromium-family browser's ``--app`` window
(Edge ships with Windows; Chrome or Chromium elsewhere), with its own
profile so the window is its own process: closing it stops the server,
which is what makes this behave like an application rather than a server
someone has to remember to kill.  Without such a browser, or with
``--tab``, the page opens in the default browser and the server runs until
Ctrl+C.  A native window (Qt WebEngine) was considered and rejected: it
would multiply the size of the Windows exe for the same page.

Run: ``polartx-gui`` / ``polartx-web`` (or ``python -m polartx.webhost``).
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

WWW = Path(__file__).resolve().parent / "webui" / "www"

_CALL_LOCK = threading.Lock()


def _rtl_export(outdir: str | None = None, n_bits: int = 10, n_thermo: int = 7,
                with_dpd: bool = True, verify: bool = True) -> dict:
    """Emit the digital datapath + Verilog-AMS PA model and, with iverilog
    installed, check every block bit-true.  Desktop only: the output is a
    directory of Verilog and the checker shells out to iverilog, neither of
    which means anything on a phone (appbridge's docstring records that)."""
    import tempfile

    from .guiutil import run_rtl_export
    out = outdir or str(Path(tempfile.gettempdir()) / "polartx_rtl")
    return run_rtl_export(out, n_bits=int(n_bits),
                          n_thermo=min(int(n_thermo), int(n_bits)),
                          with_dpd=bool(with_dpd), verify=bool(verify))


#: methods only this host answers; the page shows their tabs only when it
#: runs here (no injected window.host), and the parity test checks both
#: directions for these as it does for appbridge's
DESKTOP_METHODS = {"rtl_export": _rtl_export}


def _desktop_call(method: str, args_json: str) -> str:
    """Same envelope as appbridge.call: never raises, errors come back
    in-band."""
    import traceback

    from .appbridge import _clean
    try:
        args = json.loads(args_json or "{}")
        result = DESKTOP_METHODS[method](**args)
        return json.dumps({"ok": True, "result": _clean(result)})
    except Exception as exc:                     # noqa: BLE001 (in-band error)
        return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()})


class _Handler(BaseHTTPRequestHandler):
    server_version = "polartx-webhost"

    def log_message(self, fmt, *args):          # quiet: this is a desktop app
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                           # noqa: N802 (stdlib name)
        rel = self.path.split("?", 1)[0].lstrip("/") or "index.html"
        target = (WWW / rel).resolve()
        # only files inside www: no "../" walk out of it
        if WWW.resolve() not in target.parents or not target.is_file():
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)

    def do_POST(self):                          # noqa: N802 (stdlib name)
        if self.path.split("?", 1)[0] != "/call":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        from . import appbridge
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
            method = str(req.get("method", ""))
            args = json.dumps(req.get("args") or {})
        except (ValueError, TypeError) as exc:
            body = json.dumps({"ok": False, "error": f"bad request: {exc}"})
        else:
            with _CALL_LOCK:
                body = (_desktop_call(method, args) if method in DESKTOP_METHODS
                        else appbridge.call(method, args))
        self._send(200, body.encode("utf-8"), "application/json")


class _Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second socket bind a port that is
    # already listening, so a second launch would silently share 8765 with
    # the first and either could answer.  Elsewhere it only skips TIME_WAIT.
    allow_reuse_address = sys.platform != "win32"


def make_server(port: int = 0, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """A server bound to ``host:port`` (0 picks a free port; read it back
    from ``server.server_address``).  Not started: call ``serve_forever``."""
    return _Server((host, port), _Handler)


#: names a Chromium-family browser goes by on PATH, preferred first
_BROWSER_NAMES = ("msedge", "microsoft-edge", "microsoft-edge-stable",
                  "google-chrome", "google-chrome-stable", "chrome",
                  "chromium", "chromium-browser", "brave-browser")


def find_app_browser() -> str | None:
    """A browser that can open a chromeless ``--app`` window, or None.

    ``POLARTX_BROWSER`` overrides the search.  Windows and macOS do not put
    their browsers on PATH, so the usual install locations are tried too.
    """
    env = os.environ.get("POLARTX_BROWSER")
    if env:
        return env
    for name in _BROWSER_NAMES:
        found = shutil.which(name)
        if found:
            return found
    places: list[Path] = []
    if sys.platform == "win32":
        for var in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
            base = os.environ.get(var)
            if base:
                places += [Path(base, "Microsoft", "Edge", "Application", "msedge.exe"),
                           Path(base, "Google", "Chrome", "Application", "chrome.exe")]
    elif sys.platform == "darwin":
        for app in ("Microsoft Edge", "Google Chrome", "Chromium", "Brave Browser"):
            places.append(Path("/Applications", f"{app}.app", "Contents", "MacOS", app))
    return next((str(p) for p in places if p.is_file()), None)


def _profile_dir() -> Path:
    """The app window's own browser profile.  Persistent, because the page
    keeps its run history in localStorage; separate from the user's normal
    profile, because a browser already running with that one would take the
    window over and exit at once, and the server would never learn when the
    window closes."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    return Path(base or Path.home() / ".cache", "polartx", "app-window")


def app_window_command(browser: str, url: str, profile: Path) -> list[str]:
    return [browser, f"--app={url}", f"--user-data-dir={profile}",
            "--no-first-run", "--no-default-browser-check",
            "--window-size=1440,920"]


#: a window process that exits sooner than this handed the URL to a browser
#: already running on the same profile (a second launch while an older
#: window is open); the window the user sees then belongs to that process
HANDOFF_S = 3.0


def _stop_with_window(srv: Any, proc: Any, handoff_s: float | None = None) -> bool:
    """Wait for the app window to close, then stop the server.  Returns
    whether it stopped it (False after a hand-off: there is no window of
    ours left to wait for, so the server runs until Ctrl+C)."""
    limit = HANDOFF_S if handoff_s is None else handoff_s
    t0 = time.monotonic()
    proc.wait()
    if time.monotonic() - t0 < limit:
        print("the window was handed to a browser that was already running; "
              "the server keeps running until Ctrl+C")
        return False
    srv.shutdown()
    return True


def _open_window(url: str, tab: bool) -> subprocess.Popen | None:
    """Open the page.  Returns the app window's process when there is one."""
    browser = None if tab else find_app_browser()
    if browser:
        cmd = app_window_command(browser, url, _profile_dir())
        try:
            return subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        except OSError as exc:
            print(f"could not start {browser} ({exc}); opening a browser tab")
    webbrowser.open(url)
    return None


def _is_polartx(url: str) -> bool:
    """Whether a polartx host already answers at ``url``."""
    req = urllib.request.Request(
        url + "call", data=json.dumps({"method": "version"}).encode(),
        headers={"Content-Type": "application/json"})
    # no proxy for loopback: an HTTP(S)_PROXY in the environment would
    # otherwise carry this request off the machine
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=3) as r:
            return bool(json.loads(r.read()).get("ok"))
    except (OSError, ValueError):
        return False


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="polartx-gui",
                                 description="polartx workbench (the one page, "
                                             "served locally)")
    ap.add_argument("--port", type=int, default=8765,
                    help="local port (default 8765; 0 picks a free one -- "
                         "the page's run history is kept per port)")
    ap.add_argument("--tab", action="store_true",
                    help="open a normal browser tab instead of an app window")
    ap.add_argument("--no-browser", action="store_true",
                    help="serve only; open nothing (POLARTX_NO_BROWSER=1 "
                         "does the same, for the packaged exe)")
    a = ap.parse_args(argv)
    no_browser = a.no_browser or bool(os.environ.get("POLARTX_NO_BROWSER"))
    try:
        srv = make_server(a.port)
    except OSError as exc:
        url = f"http://127.0.0.1:{a.port}/"
        if not _is_polartx(url):
            raise SystemExit(f"port {a.port} is in use ({exc}); "
                             f"try --port 0") from exc
        # a second launch: the first one's server is the one to show
        print(f"polartx is already running at {url}")
        if not no_browser:
            _open_window(url, a.tab)
        return
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    proc = None if no_browser else _open_window(url, a.tab)
    if proc is not None:
        print(f"polartx workbench at {url}  (close its window to stop)")
        threading.Thread(target=_stop_with_window, args=(srv, proc),
                         daemon=True).start()
    else:
        print(f"polartx workbench at {url}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
