"""PySide6 desktop GUI: offscreen construction + a real chain render.

Skipped when PySide6 (or the Qt platform runtime) is unavailable —
sibling-repo convention."""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    a = QApplication.instance() or QApplication([])
    yield a


def test_main_window_builds(app):
    from polartx.guiqt.app import PAGES, MainWindow
    win = MainWindow()
    win.show()
    app.processEvents()
    n = len(PAGES)
    assert win.nav.count() == n and win.stack.count() == n
    for row in range(n):
        win.nav.setCurrentRow(row)
        app.processEvents()
    win.close()


def test_every_feature_page_is_reachable(app):
    """The newer library features (FIR/Doherty benchmark, architecture
    selector, RTL/AMS export) each have a page — the GUI must not lag the
    library."""
    from polartx.guiqt.app import PAGES
    titles = {c.title for c in PAGES}
    assert {"FIR + Doherty", "Architecture Selector",
            "RTL / AMS Export"} <= titles


def test_chain_page_renders_result(app):
    from polartx.guiqt.app import MainWindow
    from polartx.guiutil import run_chain_report
    win = MainWindow()
    page = win.pages[0]
    page.preset.setCurrentText("BLE LE-1M")
    rep = run_chain_report("BLE LE-1M", seed=1, noise=True,
                           **page._overrides("BLE LE-1M"))
    page._show(rep)
    app.processEvents()
    assert page._table is not None
    assert page.figbox._canvas is not None
    win.close()


def test_overrides_dispatch(app):
    from polartx.guiqt.app import MainWindow
    win = MainWindow()
    page = win.pages[0]
    page.dtc_bits.setValue(9)
    assert page._overrides("WiFi 160 MHz")["n_bits"] == 9
    page.dp_err_pct.setValue(5.0)
    assert page._overrides("BLE LE-1M")["dp_gain"] == pytest.approx(1.05)
    assert "dpd" in page._overrides("LTE 20 MHz")
    win.close()


@pytest.mark.parametrize("cls_name", ["FIRDohertyPage", "SelectorPage",
                                      "RTLPage"])
def test_new_pages_construct_and_render(app, cls_name, tmp_path):
    """Each new page builds its widgets and its _show() accepts the report
    its guiutil function actually returns (the wiring, not just the import)."""
    import polartx.guiqt.pages as P
    from polartx import guiutil as G
    page = getattr(P, cls_name)()
    if cls_name == "SelectorPage":
        rep = G.run_selector_report(bw_hz=80e6, evm_db_max=-35.0)
        page._show(rep)
        app.processEvents()
        # this page reports into a label + crossover chart, not a table
        assert "recommend" in page.rec.text()
        assert page.figbox._canvas is not None
        return
    if cls_name == "RTLPage":
        page._outdir = str(tmp_path)
        rep = G.run_rtl_export(str(tmp_path), n_bits=8, n_thermo=5,
                               verify=False)
        page._show(rep)
        app.processEvents()
        assert page._table is not None and "files in" in page.files.text()
        return
    rep = G.run_combiner_report(n_way=2)
    page._show(rep)
    app.processEvents()
    assert page._table is not None and page.figbox._canvas is not None
