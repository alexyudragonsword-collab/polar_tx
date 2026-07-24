"""Shared Qt building blocks (pllsim.guiqt conventions): worker thread,
page base with single-flight async runs, matplotlib canvas embedding."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")            # figures are re-parented onto Qt canvases
import matplotlib.pyplot as plt  # noqa: E402,F401
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QMessageBox, QPushButton, QSizePolicy,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)


class Worker(QThread):
    """Run a plain callable off the UI thread."""

    done = Signal(object)
    fail = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn())
        except Exception as exc:            # surfaced in a message box
            self.fail.emit(f"{type(exc).__name__}: {exc}")


class Page(QWidget):
    """Base page: single-flight worker + busy buttons + error dialog."""

    title = "page"

    def __init__(self):
        super().__init__()
        self._worker = None
        self._busy: list[QPushButton] = []

    def run_async(self, fn, on_done, *buttons: QPushButton):
        if self._worker is not None and self._worker.isRunning():
            return
        self._busy = list(buttons)
        for b in self._busy:
            b.setEnabled(False)
        self._worker = Worker(fn, self)
        self._worker.done.connect(lambda r: self._finish(on_done, r))
        self._worker.fail.connect(self._error)
        self._worker.start()

    def _finish(self, on_done, result):
        for b in self._busy:
            b.setEnabled(True)
        on_done(result)

    def _error(self, msg: str):
        for b in self._busy:
            b.setEnabled(True)
        QMessageBox.critical(self, "polartx", msg)


class FigureBox(QWidget):
    """Holds one matplotlib figure, replaced on every run."""

    def __init__(self):
        super().__init__()
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._canvas = None

    def set_figure(self, fig):
        if self._canvas is not None:
            self._lay.removeWidget(self._canvas)
            self._canvas.deleteLater()
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        self._lay.addWidget(self._canvas)
        self._canvas.draw_idle()


def metrics_table(metrics: dict) -> QTableWidget:
    t = QTableWidget(len(metrics), 2)
    t.setHorizontalHeaderLabels(["metric", "value"])
    for i, (k, v) in enumerate(metrics.items()):
        t.setItem(i, 0, QTableWidgetItem(str(k)))
        t.setItem(i, 1, QTableWidgetItem(str(v)))
    t.resizeColumnsToContents()
    t.setMaximumHeight(30 * (len(metrics) + 1) + 10)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    return t
