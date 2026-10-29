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

Run: ``polartx-web`` (or ``python -m polartx.webhost``).
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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


def make_server(port: int = 0, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """A server bound to ``host:port`` (0 picks a free port; read it back
    from ``server.server_address``).  Not started: call ``serve_forever``."""
    return ThreadingHTTPServer((host, port), _Handler)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="polartx-web",
                                 description="polartx workbench in a browser")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true",
                    help="serve only; do not open a browser window")
    a = ap.parse_args(argv)
    srv = make_server(a.port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"polartx workbench at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
