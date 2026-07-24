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
    from polartx.guiqt.app import MainWindow
    win = MainWindow()
    win.show()
    app.processEvents()
    assert win.nav.count() == 3 and win.stack.count() == 3
    for row in range(3):
        win.nav.setCurrentRow(row)
        app.processEvents()
    win.close()


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
