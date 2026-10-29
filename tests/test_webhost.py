"""The desktop host's lifecycle: the page opens in a browser app window, and
closing that window stops the server.

``tests/test_webui.py`` drives the page itself through this host; this file
is about the process around it -- which window opens, when the server stops,
what a second launch does.  A real browser is not needed: the window is a
child process, so a small Python script stands in for it and does what the
browser would (fetch the page, call the bridge, exit when "closed").
"""
from __future__ import annotations

import importlib
import json
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from polartx import webhost

ROOT = Path(__file__).resolve().parent.parent

# what the stand-in browser runs: load the page and one bridge call the way
# the real window would, record what came back, then "close"
_FAKE_BROWSER = r"""
import json, sys, urllib.request
url, out = sys.argv[1], sys.argv[2]
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
page = op.open(url, timeout=30).read().decode()
req = urllib.request.Request(url + "call",
                             data=json.dumps({"method": "version"}).encode())
ver = json.loads(op.open(req, timeout=60).read())
open(out, "w").write(json.dumps({"page": "<canvas" in page or "<html" in page,
                                 "ok": ver["ok"]}))
"""


def _run_main(argv, timeout=90.0):
    """main() in a thread; returns (finished, exception)."""
    box: dict = {}

    def target():
        try:
            webhost.main(argv)
        except BaseException as exc:            # noqa: BLE001 (reported below)
            box["exc"] = exc
    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout)
    return not th.is_alive(), box.get("exc")


def test_the_app_window_command_is_a_chromeless_window_on_its_own_profile():
    cmd = webhost.app_window_command("edge", "http://127.0.0.1:1/", Path("p"))
    assert cmd[0] == "edge"
    assert "--app=http://127.0.0.1:1/" in cmd
    # its own profile is what makes the window its own process -- the
    # property the server's shutdown depends on
    assert f"--user-data-dir={Path('p')}" in cmd


def test_polartx_browser_overrides_the_search(monkeypatch):
    monkeypatch.setenv("POLARTX_BROWSER", "/somewhere/chromium")
    assert webhost.find_app_browser() == "/somewhere/chromium"


def test_closing_the_window_stops_the_server(monkeypatch, tmp_path):
    out = tmp_path / "seen.json"
    monkeypatch.setattr(webhost, "find_app_browser", lambda: sys.executable)
    monkeypatch.setattr(webhost, "app_window_command",
                        lambda b, url, p: [b, "-c", _FAKE_BROWSER, url, str(out)])
    monkeypatch.setattr(webhost, "HANDOFF_S", 0.0)
    monkeypatch.delenv("POLARTX_NO_BROWSER", raising=False)
    finished, exc = _run_main(["--port", "0"])
    assert exc is None
    assert finished, "the server kept running after its window closed"
    seen = json.loads(out.read_text())
    assert seen == {"page": True, "ok": True}


class _Srv:
    def __init__(self):
        self.stopped = False

    def shutdown(self):
        self.stopped = True


def test_a_window_handed_to_a_running_browser_does_not_stop_the_server():
    # the browser process exits at once when another one already owns the
    # profile; the window the user sees is not ours to wait for
    quick = subprocess.Popen([sys.executable, "-c", "pass"])
    srv = _Srv()
    assert webhost._stop_with_window(srv, quick, handoff_s=30.0) is False
    assert not srv.stopped
    slow = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.3)"])
    srv = _Srv()
    assert webhost._stop_with_window(srv, slow, handoff_s=0.0) is True
    assert srv.stopped


def test_a_second_launch_shows_the_running_host(monkeypatch):
    srv = webhost.make_server(0)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    opened = []
    monkeypatch.setattr(webhost, "_open_window",
                        lambda url, tab: opened.append(url))
    monkeypatch.delenv("POLARTX_NO_BROWSER", raising=False)
    try:
        finished, exc = _run_main(["--port", str(port)], timeout=30)
    finally:
        srv.shutdown()
        srv.server_close()
    assert finished and exc is None
    assert opened == [f"http://127.0.0.1:{port}/"]


def test_a_port_held_by_something_else_is_an_error():
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        t0 = time.monotonic()
        finished, exc = _run_main(["--port", str(port), "--no-browser"],
                                  timeout=30)
    finally:
        holder.close()
    assert finished
    assert isinstance(exc, SystemExit) and "in use" in str(exc)
    assert time.monotonic() - t0 < 30


def test_the_server_refuses_a_port_that_is_already_serving():
    # on Windows this is the reason _Server drops SO_REUSEADDR: with it, the
    # second bind succeeds and two hosts share one port
    a = webhost.make_server(0)
    try:
        with pytest.raises(OSError):
            webhost.make_server(a.server_address[1])
    finally:
        a.server_close()


def test_every_console_command_resolves_to_a_callable():
    # polartx-gui pointed at polartx.guiqt.app:main until the Qt front end
    # retired; an entry point naming a deleted module installs cleanly and
    # fails only when someone types the command
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    scripts = dict(re.findall(r'^(polartx-[a-z]+)\s*=\s*"([^"]+)"', text, re.M))
    assert {"polartx-gui", "polartx-web"} <= set(scripts)
    for target in scripts.values():
        mod, fn = target.split(":")
        assert callable(getattr(importlib.import_module(mod), fn)), target
