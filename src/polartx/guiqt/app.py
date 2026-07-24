"""polartx native desktop workbench (PySide6).

Run: polartx-gui  (or python -m polartx.guiqt)
Sidebar navigation over the feature pages; all computation in
polartx.guiutil on worker threads (pllsim.guiqt conventions).
"""
from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import (QApplication, QHBoxLayout, QListWidget,
                               QMainWindow, QStackedWidget, QWidget)

from .pages import CalibrationPage, ChainPage, MonteCarloPage

PAGES = [ChainPage, CalibrationPage, MonteCarloPage]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("polartx — digital polar TX workbench")
        self.resize(1240, 820)
        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        self.nav = QListWidget()
        self.nav.setMaximumWidth(200)
        self.stack = QStackedWidget()
        self.pages = []
        for cls in PAGES:
            page = cls()
            self.pages.append(page)
            self.nav.addItem(cls.title)
            self.stack.addWidget(page)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)
        lay.addWidget(self.nav)
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(central)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("polartx")
    win = MainWindow()
    win.show()
    if os.environ.get("POLARTX_SMOKE"):
        # CI smoke: open, render, exit clean after a few seconds
        from PySide6.QtCore import QTimer
        QTimer.singleShot(3000, app.quit)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
