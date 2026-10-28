"""Streamlit web GUI: every page renders and every button actually runs.

Uses streamlit's AppTest, which executes the page script for real (not a
syntax check) — the web frontend gets the same regression protection the
Qt one has.  Skipped when streamlit is unavailable.
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = sorted(glob.glob(os.path.join(REPO, "gui", "pages", "*.py")))
HOME = os.path.join(REPO, "gui", "Home.py")


def _run(path):
    at = AppTest.from_file(path, default_timeout=900)
    at.run()
    assert not at.exception, f"{os.path.basename(path)}: {at.exception}"
    return at


def test_home_renders():
    _run(HOME)


@pytest.mark.parametrize("path", PAGES, ids=lambda p: os.path.basename(p))
def test_page_renders(path):
    _run(path)


@pytest.mark.parametrize("path", PAGES, ids=lambda p: os.path.basename(p))
def test_every_button_runs(path):
    """Click each button on a fresh script run: catches wiring bugs that a
    render-only check (or py_compile) sails straight past."""
    at = _run(path)
    for btn in list(at.button):
        fresh = _run(path)
        target = [b for b in fresh.button if b.label == btn.label][0]
        target.click().run()
        assert not fresh.exception, \
            f"{os.path.basename(path)} [{btn.label}]: {fresh.exception}"


def test_newest_features_have_web_pages():
    """The web frontend must not lag the library: the FIR/Doherty
    benchmark, the architecture selector and the RTL/AMS export each have
    a page."""
    names = {os.path.basename(p) for p in PAGES}
    assert any("FIR" in n for n in names)
    assert any("Selector" in n for n in names)
    assert any("RTL" in n for n in names)


def test_chain_page_offers_every_registered_preset():
    at = _run(os.path.join(REPO, "gui", "pages", "1_Chain_Workbench.py"))
    from polartx.guiutil import PRESETS
    opts = {str(o) for sb in at.selectbox for o in sb.options}
    missing = set(PRESETS) - opts
    assert not missing, f"presets unreachable from the web GUI: {missing}"


def test_chain_page_offers_only_the_knobs_the_preset_honours():
    """The DTC resolution slider used to be sent to the RF-DAC chain as
    its DAC resolution, and the skew slider shown on chains it cannot
    act on; each preset now shows exactly the knobs it honours."""
    from polartx.guiutil import DEFAULT_PRESET
    at = _run(os.path.join(REPO, "gui", "pages", "1_Chain_Workbench.py"))
    assert at.selectbox[0].value == DEFAULT_PRESET
    labels = lambda: {s.label for s in at.slider}            # noqa: E731
    assert {"DTC bits", "AM/PM skew [ns]"} <= labels()
    at.selectbox[0].select("WiFi 160 MHz (RF-DAC)").run()
    assert not at.exception
    assert "DTC bits" not in labels() and "AM/PM skew [ns]" not in labels()
    at.selectbox[0].select("BLE LE-1M").run()
    assert "AM/PM skew [ns]" not in labels()
    assert "loop BW [kHz]" in labels()


def test_home_names_every_page_and_topology():
    """The landing page described two architectures and pointed at
    ex01-ex07 long after there were four topologies and 21 examples.
    It must at least name every page that exists and each topology."""
    text = open(HOME, encoding="utf-8").read()
    for path in PAGES:
        title = os.path.basename(path)[2:-3].replace("_", " ")
        assert title in text, f"Home does not mention the {title!r} page"
    for topo in ("ADPLL", "DTC", "Outphasing", "RF-DAC"):
        assert topo in text, topo
    assert "ex01" not in text


def test_chain_run_states_its_conventions():
    at = _run(os.path.join(REPO, "gui", "pages", "1_Chain_Workbench.py"))
    at.button[0].click().run()
    assert not at.exception
    assert any("scalar equalization" in c.value for c in at.caption)
