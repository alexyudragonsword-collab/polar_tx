"""The desktop and browser host for the polartx page.

The phone shows ``webui/www`` in a WebView and reaches Python through a
JavaScript interface; this module gives a laptop the same page and the same
bridge over a local HTTP server, so all three hosts run one front end:

* ``GET /`` and ``GET /<file>`` serve ``webui/www`` (the files the APK
  bundles as its assets);
* ``POST /call`` with ``{"method": ..., "args": {...}}`` answers with
  exactly what :func:`polartx.appbridge.call` returns — the same envelope
  the phone receives.

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
                body = appbridge.call(method, args)
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
