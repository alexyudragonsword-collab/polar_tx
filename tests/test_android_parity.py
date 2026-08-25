"""The Android page and the bridge must not drift apart.

A real browser check would need Playwright and minutes; these are the parts
checkable from text alone, in every CI run, at no cost — and they are the
parts that actually go wrong.  This repository has already shipped both
failure modes on other surfaces: a library feature the GUI could not reach,
and a page element the code addressed by an id that had been renamed.
"""
import re
from pathlib import Path

import pytest

from polartx import appbridge

WWW = Path(__file__).resolve().parents[1] / "android/app/src/main/assets/www"
APP_JS = WWW / "app.js"
INDEX = WWW / "index.html"
CSS = WWW / "style.css"


def test_every_bridge_method_is_called_by_the_page():
    """A bridge method with no caller is half a feature: it has bridge
    tests, looks covered, and does nothing for the user.

    Mutation check: add an entry to appbridge._METHODS and this goes red
    until app.js calls it; delete a call() from app.js and it goes red too.
    """
    js = APP_JS.read_text()
    unused = [m for m in appbridge._METHODS if f'"{m}"' not in js]
    assert not unused, (
        f"appbridge exposes {unused} that app.js never calls — wire the "
        "render in the same change, or drop the method")


def test_the_page_calls_nothing_the_bridge_does_not_expose():
    """The other direction: a typo'd method name fails only at runtime, in
    a WebView, where the user sees an error card and nobody sees a log."""
    js = APP_JS.read_text()
    called = set(re.findall(r'\bcall\(\s*"([a-z_]+)"', js))
    missing = sorted(called - set(appbridge._METHODS))
    assert not missing, f"app.js calls {missing}, which appbridge has no entry for"


def test_every_element_app_js_drives_exists_in_the_page():
    """app.js addresses the DOM by id; a renamed id in index.html turns a
    live control into silence with no error anywhere.

    Derived from app.js rather than listed by hand — a hardcoded list is
    the thing that stops tracking reality first.
    """
    js = APP_JS.read_text()
    html = INDEX.read_text()
    ids = set(re.findall(r'\$\("([a-z0-9-]+)"\)', js))
    missing = sorted(i for i in ids if f'id="{i}"' not in html)
    assert not missing, f"app.js drives {missing}, which index.html does not define"


def test_every_tab_button_has_a_panel_and_vice_versa():
    html = INDEX.read_text()
    tabs = set(re.findall(r'<button data-tab="([a-z]+)"', html))
    panels = set(re.findall(r'<div id="tab-([a-z]+)"', html))
    assert tabs == panels, f"tabs {tabs} but panels {panels}"
    assert tabs, "no tabs at all — the regex stopped matching"


def test_bilingual_strings_carry_both_languages():
    """The language toggle reads dataset[lang]; an element with data-zh and
    no data-en goes blank in English rather than falling back."""
    html = INDEX.read_text()
    tags = re.findall(r'<[^>]*\bdata-zh=[^>]*>', html)
    assert tags, "the bilingual regex matched nothing — it is not checking anything"
    missing = [t[:70] for t in tags if "data-en=" not in t]
    assert not missing, f"data-zh without data-en: {missing}"


def test_the_busy_overlay_cannot_swallow_taps_while_hidden():
    """An author `display` beats the UA's [hidden]{display:none}, leaving an
    invisible overlay that eats every tap — the app looks fine and responds
    to nothing.  The rule below is the fix, and it is easy to lose in a
    stylesheet edit."""
    css = CSS.read_text()
    assert "#busy[hidden]" in css and "display: none" in css


def test_the_overlay_is_released_on_the_error_path():
    """An early return that skips busy(false) leaves the app permanently
    unresponsive with nothing on screen to explain it, so the release has
    to sit in `finally`."""
    js = APP_JS.read_text()
    body = js[js.index("async function run("):]
    body = body[:body.index("\n}\n")]
    assert "finally" in body and "busy(false)" in body


def test_the_page_accounts_for_notches_and_gesture_bars():
    """Device-only symptom: without these the first control sits under the
    status bar, and no emulator screenshot in CI would show it."""
    assert "viewport-fit=cover" in INDEX.read_text()
    assert "env(safe-area-inset-top)" in CSS.read_text()


def test_the_app_declares_no_permissions():
    """A tool of this shape computes locally and fetches nothing.  If a
    change ever wants INTERNET, this test is where the conversation should
    start — it converts a self-contained instrument into one that can send
    whatever the user typed somewhere else."""
    manifest = (WWW.parents[1] / "AndroidManifest.xml").read_text()
    # strip comments first: the manifest explains this policy in prose, and
    # a naive substring search matches its own explanation
    body = re.sub(r"<!--.*?-->", "", manifest, flags=re.S)
    declared = re.findall(r"<uses-permission[^>]*android:name=\"([^\"]+)\"", body)
    assert not declared, f"the app declares {declared}"


@pytest.mark.parametrize("method", sorted(appbridge._METHODS))
def test_each_method_has_a_visible_output_target(method):
    """Beyond being called: the result has to land somewhere on screen.
    `version` and `list_presets` render into the header and the preset
    select during boot; the rest write into a card."""
    js = APP_JS.read_text()
    call_site = js[js.index(f'"{method}"'):][:900]
    assert re.search(r'\$\("[a-z0-9-]+"\)|run\(\s*"[a-z0-9-]+"', call_site), \
        f"{method} is called but nothing renders its result"
