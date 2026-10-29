"""The one page, driven in a real browser against the real bridge.

The desktop and browser host (polartx.webhost) serves the same files the
APK bundles, so this exercises what every host shows: boot, the workbench
recomputing on a control change, controls generated from the specs, the
run-time warning path, the language toggle, the phone layout and one of
the older run-button tabs.  Assertions are on physics and wiring, not on
pixels or snapshot numbers.

Needs Playwright and a Chromium.  Locally it skips without them; CI's
test-gui job sets POLARTX_REQUIRE_BROWSER=1, which turns that skip into a
failure — a green run that tested nothing looks exactly like a green run
that tested something.
"""
import json
import os
import threading
import urllib.error
import urllib.request

import pytest

from polartx.webhost import make_server


@pytest.fixture(scope="module")
def server():
    srv = make_server(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()
    srv.server_close()


# ------------------------------------------------- the host, no browser
def _post(url, payload: bytes) -> dict:
    req = urllib.request.Request(url + "call", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def test_host_serves_the_page_and_the_bridge(server):
    with urllib.request.urlopen(server, timeout=10) as r:
        assert r.status == 200
        assert "charset=utf-8" in r.headers["Content-Type"]
        assert b"app.js" in r.read()
    for name in ("app.js", "charts.js", "style.css"):
        with urllib.request.urlopen(server + name, timeout=10) as r:
            assert r.status == 200
    ok = _post(server, json.dumps({"method": "version", "args": {}}).encode())
    assert ok["ok"] and ok["result"]["lib"]
    # the same envelope as the phone gets, errors included
    bad = _post(server, json.dumps({"method": "nope"}).encode())
    assert bad["ok"] is False and "no such method" in bad["error"]
    garbled = _post(server, b"{not json")
    assert garbled["ok"] is False


def test_host_serves_nothing_outside_the_page(server):
    for path in ("../webhost.py", "..%2Fwebhost.py", "../../../pyproject.toml"):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(server + path, timeout=10)
        assert e.value.code == 404


# ------------------------------------------------------ the browser
def _browser():
    """A Chromium, or a skip (a failure where CI requires one)."""
    required = os.environ.get("POLARTX_REQUIRE_BROWSER") == "1"
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        if required:
            raise
        pytest.skip("playwright not installed")
    pw = sync_playwright().start()
    exe = os.environ.get("POLARTX_CHROMIUM")
    if not exe and os.path.exists("/opt/pw-browsers/chromium"):
        exe = "/opt/pw-browsers/chromium"
    try:
        b = pw.chromium.launch(executable_path=exe) if exe else pw.chromium.launch()
    except Exception as exc:                    # no browser binary here
        pw.stop()
        if required:
            raise
        pytest.skip(f"no Chromium: {exc}")
    return pw, b


@pytest.fixture(scope="module")
def browser():
    pw, b = _browser()
    yield b
    b.close()
    pw.stop()


def _open(browser, server, **ctx):
    page = browser.new_page(**ctx)
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(server)
    page.wait_for_selector("#app:not([hidden])", timeout=60_000)
    page.wait_for_selector("#bench-cards .mcard", timeout=60_000)
    page._errors = errors                        # noqa: SLF001
    return page


def _card(page, label: str) -> str:
    """The value text of the metric card with this label."""
    return page.locator("#bench-cards .mcard", has_text=label).first \
        .locator(".v, .pill").first.inner_text()


def _settle(page):
    """Wait until the workbench has no run in flight and nothing queued."""
    page.wait_for_function(
        "!document.getElementById('tab-bench').classList.contains('stale')",
        timeout=60_000)


def _set_range(page, knob_id: str, value: float):
    page.eval_on_selector(
        f"#knob-{knob_id}",
        "(e, v) => { e.value = v; e.dispatchEvent(new Event('input')); }",
        value)
    page.wait_for_timeout(300)
    _settle(page)


def test_workbench_recomputes_when_a_control_moves(browser, server):
    """No Run button: moving AM/PM skew on WiFi 160 MHz re-runs the chain,
    and the physics shows — the mask goes first, EVM drops by >10 dB."""
    page = _open(browser, server, viewport={"width": 1366, "height": 860})
    assert page.input_value("#bench-preset") == "WiFi 160 MHz"
    assert page.locator("#knob-dtc_bits").count() == 1
    evm0 = float(_card(page, "EVM").split()[0])
    assert _card(page, "频谱 mask") == "通过"
    _set_range(page, "skew_ns", 0.3)
    evm1 = float(_card(page, "EVM").split()[0])
    assert evm1 > evm0 + 10.0, (evm0, evm1)
    assert _card(page, "频谱 mask") == "失败"
    # the canvas really drew the report's data
    ink = page.evaluate("""() => { const c = document.getElementById('bench-canvas');
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        let n = 0; for (let i = 3; i < d.length; i += 4) n += d[i] > 0; return n; }""")
    assert ink > 1000
    box = page.locator("#bench-canvas").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    assert page.inner_text("#bench-readout").strip()
    assert not page._errors                      # noqa: SLF001
    page.close()


def test_controls_follow_the_preset_spec(browser, server):
    """The form is generated from guispec: the RF-DAC offers its own DAC
    resolution at its own default (12) and no skew; the LTE event engine
    surfaces its validity-domain warning on the page."""
    page = _open(browser, server, viewport={"width": 1366, "height": 860})
    page.select_option("#bench-preset", "WiFi 160 MHz (RF-DAC)")
    page.wait_for_selector("#knob-dac_bits")
    _settle(page)
    assert page.input_value("#knob-dac_bits") == "12"
    assert page.locator("#knob-skew_ns").count() == 0
    assert page.locator("#knob-dtc_bits").count() == 0
    page.select_option("#bench-preset", "LTE 20 MHz")
    page.wait_for_selector("#knob-mode")
    _settle(page)
    page.select_option("#knob-mode", "event")
    page.wait_for_timeout(300)
    _settle(page)
    page.wait_for_selector("#bench-notes .note", timeout=60_000)
    assert "validity domain" in page.inner_text("#bench-notes")
    page.click("#bench-reset")
    page.wait_for_timeout(300)
    _settle(page)
    assert page.input_value("#knob-mode") == "response"
    assert page.inner_text("#bench-notes").strip() == ""
    page.close()


def test_language_toggle_keeps_every_control(browser, server):
    page = _open(browser, server, viewport={"width": 1366, "height": 860})
    n = page.locator("#bench-form input, #bench-form select").count()
    page.click("#lang")
    assert page.inner_text("#nav").split()[0] == "Workbench"
    assert page.locator("#bench-form input, #bench-form select").count() == n
    assert "EVM" in page.inner_text("#bench-cards")
    page.close()


def test_phone_layout_has_no_sideways_scroll_and_the_old_tabs_run(browser, server):
    page = _open(browser, server, viewport={"width": 390, "height": 844},
                 is_mobile=True, has_touch=True)
    assert page.evaluate("document.documentElement.scrollWidth") <= 390
    assert page.is_visible("#nav-toggle")
    page.click("#nav-toggle")
    page.click("#nav button[data-tab=combiner]")
    assert page.is_visible("#run-combiner")
    page.click("#run-combiner")
    page.wait_for_selector("#cb-out img", timeout=60_000)
    assert page.is_hidden("#busy")
    assert not page._errors                      # noqa: SLF001
    page.close()


def test_baseline_sweep_and_history(browser, server):
    """Pin a baseline, move a control: the cards show the change against it
    (EVM worse, coloured as worse) and the plot overlays it.  A sweep adds
    a plot tab whose points carry the mask verdicts.  The history restores
    an earlier setting."""
    page = _open(browser, server, viewport={"width": 1366, "height": 860})
    _settle(page)
    page.click("#bench-pin")
    assert page.is_visible("#bench-unpin")
    _set_range(page, "skew_ns", 0.3)
    delta = page.locator("#bench-cards .mcard", has_text="EVM").first \
        .locator(".delta")
    assert float(delta.inner_text()) > 10.0              # dB worse than baseline
    assert "worse" in delta.get_attribute("class")
    # the sweep: one control across its range, as an extra plot tab
    page.click("#sweep-dtc_bits")
    page.wait_for_function("document.getElementById('busy').hidden",
                           timeout=120_000)
    tabs = page.locator("#bench-plot-tabs button")
    assert "DTC" in tabs.last.inner_text()
    assert tabs.last.get_attribute("aria-selected") == "true"
    assert page.is_visible("#sweep-metric")
    assert page.locator("#sweep-metric option").count() >= 3
    # history: the newest first; restoring the baseline setting undoes skew
    chips = page.locator("#bench-history button")
    assert chips.count() >= 2 and "0.3" in chips.first.inner_text()
    chips.last.click()
    page.wait_for_timeout(300)
    _settle(page)
    assert page.input_value("#knob-skew_ns") == "0"
    assert _card(page, "频谱 mask") == "通过"
    page.click("#bench-unpin")
    assert page.locator("#bench-cards .delta").count() == 0
    # ids are unique: the comparison's controls do not shadow the workbench's
    dup = page.evaluate("""() => { const seen = {}, dup = [];
        document.querySelectorAll('[id]').forEach(e => {
          if (seen[e.id]) dup.push(e.id); seen[e.id] = 1; }); return dup; }""")
    assert dup == []
    assert not page._errors                      # noqa: SLF001
    page.close()


def test_topologies_side_by_side(browser, server):
    """Three topologies on one burst: a skew hurts the polar chain only."""
    page = _open(browser, server, viewport={"width": 1366, "height": 860})
    page.click("#nav button[data-tab=compare]")
    page.eval_on_selector(
        "#cmp-knob-skew_ns",
        "e => { e.value = 0.5; e.dispatchEvent(new Event('input')); }")
    page.click("#run-compare")
    page.wait_for_selector("#cmp-out table", timeout=120_000)
    head = page.locator("#cmp-out thead th").all_inner_texts()
    assert len(head) == 4 and "RF-DAC" in head[3]
    evm = page.locator("#cmp-out tbody tr", has_text="EVM").first \
        .locator("td.n").all_inner_texts()
    polar, outph, rfdac = (float(v) for v in evm)
    assert polar > rfdac + 10.0 and polar > outph + 10.0
    assert page.is_visible("#cmp-canvas")
    assert not page._errors                      # noqa: SLF001
    page.close()
