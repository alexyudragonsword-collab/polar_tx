"""Feature pages, all computation via the headless polartx.guiutil layer."""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QSpinBox, QVBoxLayout)

from ..guiutil import PRESETS, run_chain_report, run_mc_report
from .widgets import FigureBox, Page, metrics_table


class ChainPage(Page):
    title = "Chain Workbench"

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        form_box = QGroupBox("chain")
        form = QFormLayout(form_box)
        self.preset = QComboBox()
        self.preset.addItems(PRESETS)
        self.preset.setCurrentIndex(6)
        self.seed = QSpinBox()
        self.seed.setRange(0, 9999)
        self.seed.setValue(1)
        self.noise = QCheckBox("noise on")
        self.noise.setChecked(True)
        self.dtc_bits = QSpinBox()
        self.dtc_bits.setRange(6, 14)
        self.dtc_bits.setValue(11)
        self.skew_ns = QDoubleSpinBox()
        self.skew_ns.setRange(-4.0, 4.0)
        self.skew_ns.setSingleStep(0.1)
        self.dp_err_pct = QDoubleSpinBox()
        self.dp_err_pct.setRange(-10.0, 10.0)
        self.dp_err_pct.setSingleStep(0.5)
        self.dpd = QCheckBox("polar DPD (LTE)")
        self.dpd.setChecked(True)
        form.addRow("preset", self.preset)
        form.addRow("seed", self.seed)
        form.addRow("", self.noise)
        form.addRow("DTC bits (WiFi/NR)", self.dtc_bits)
        form.addRow("AM/PM skew [ns] (WiFi/NR)", self.skew_ns)
        form.addRow("two-point gain err [%] (BLE/EDR/LTE)", self.dp_err_pct)
        form.addRow("", self.dpd)
        self.btn = QPushButton("Run")
        self.btn.clicked.connect(self._go)
        top = QHBoxLayout()
        top.addWidget(form_box, 1)
        top.addWidget(self.btn)
        lay.addLayout(top)
        self.table_slot = QVBoxLayout()
        lay.addLayout(self.table_slot)
        self.figbox = FigureBox()
        lay.addWidget(self.figbox, 1)
        self._table = None

    def _overrides(self, name: str) -> dict:
        ov = {}
        if "WiFi" in name or "NR" in name:
            ov["n_bits"] = self.dtc_bits.value()
            ov["env_skew_s"] = self.skew_ns.value() * 1e-9
        elif "LTE" in name:
            ov["dpd"] = self.dpd.isChecked()
            ov["dp_gain"] = 1.0 + self.dp_err_pct.value() / 100.0
        else:                       # BLE / EDR
            ov["dp_gain"] = 1.0 + self.dp_err_pct.value() / 100.0
        return ov

    def _go(self):
        name = self.preset.currentText()
        seed, noise = self.seed.value(), self.noise.isChecked()
        ov = self._overrides(name)
        self.run_async(
            lambda: run_chain_report(name, seed=seed, noise=noise, **ov),
            self._show, self.btn)

    def _show(self, rep):
        if self._table is not None:
            self.table_slot.removeWidget(self._table)
            self._table.deleteLater()
        self._table = metrics_table(rep["metrics"])
        self.table_slot.addWidget(self._table)
        self.figbox.set_figure(rep["fig"])


class CalibrationPage(Page):
    title = "Calibration Lab"

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)

        g1 = QGroupBox("AM/PM skew estimate (WiFi 160 MHz)")
        f1 = QFormLayout(g1)
        self.skew_in = QDoubleSpinBox()
        self.skew_in.setRange(0.0, 4.0)
        self.skew_in.setValue(2.0)
        self.skew_in.setSingleStep(0.1)
        self.skew_method = QComboBox()
        self.skew_method.addItems(["waveform align", "ACP power search"])
        self.b1 = QPushButton("estimate")
        self.b1.clicked.connect(self._skew)
        self.r1 = QLabel("—")
        f1.addRow("injected skew [ns]", self.skew_in)
        f1.addRow("estimator", self.skew_method)
        f1.addRow(self.b1, self.r1)
        lay.addWidget(g1)

        g2 = QGroupBox("two-point gain (LTE 20 MHz, offline LS)")
        f2 = QFormLayout(g2)
        self.eps_in = QDoubleSpinBox()
        self.eps_in.setRange(-8.0, 8.0)
        self.eps_in.setValue(3.0)
        self.b2 = QPushButton("estimate + correct")
        self.b2.clicked.connect(self._twopoint)
        self.r2 = QLabel("—")
        f2.addRow("injected gain error [%]", self.eps_in)
        f2.addRow(self.b2, self.r2)
        lay.addWidget(g2)

        g3 = QGroupBox("DTC gain/INL LUT cal (CW training, 2 iterations)")
        f3 = QFormLayout(g3)
        self.b3 = QPushButton("run")
        self.b3.clicked.connect(self._dtc)
        self.r3 = QLabel("—")
        f3.addRow(self.b3, self.r3)
        lay.addWidget(g3)
        lay.addStretch(1)

    def _skew(self):
        inj = self.skew_in.value() * 1e-9
        method = self.skew_method.currentText()

        def job():
            from ..cal.skew import estimate_env_skew, estimate_skew_by_acp
            from ..presets import wifi_dtc
            p = wifi_dtc(bw=160e6, env_skew_s=inj)
            wf = p.make_waveform(n_symbols=4, seed=2)
            if method == "waveform align":
                return estimate_env_skew(p.tx.run(wf, noise=False))
            return estimate_skew_by_acp(p.tx, wf)

        self.run_async(job, lambda est: self.r1.setText(
            f"estimated {est['skew_s'] * 1e9:.2f} ns "
            f"(error {abs(est['skew_s'] - inj) * 1e12:.0f} ps)"), self.b1)

    def _twopoint(self):
        eps = self.eps_in.value() / 100.0

        def job():
            from ..cal.twopoint import estimate_dp_gain_error
            from ..presets import lte20_adpll
            p = lte20_adpll(qam=64, dp_gain=1.0 + eps)
            wf = p.make_waveform(n_symbols=10, seed=0)
            r0 = p.tx.run(wf, noise=True, seed=2)
            est = estimate_dp_gain_error(p.tx.phasemod, r0.phase_cmd,
                                         r0.phase_out, r0.fs)
            p.tx.phasemod.dp_gain *= est["dp_gain_corr"]
            r1 = p.tx.run(wf, noise=True, seed=2)
            return est, r0.evm().db, r1.evm().db

        self.run_async(job, lambda t: self.r2.setText(
            f"eps_hat {100 * t[0]['eps_hat']:.2f} %, "
            f"EVM {t[1]:.1f} → {t[2]:.1f} dB"), self.b2)

    def _dtc(self):
        def job():
            from ..cal.dtc_cal import (apply_dtc_correction,
                                       fit_dtc_correction)
            from ..phasemod import DTCPhaseModulator, DTCPMConfig
            fs, n = 640e6, 1 << 16
            cw = 2 * np.pi * 5e6 * np.arange(n) / fs
            pm = DTCPhaseModulator(DTCPMConfig(
                n_bits=12, gain_error=0.01, inl_sin=(2e-3, 3, 0.0)))
            before = fit_dtc_correction(pm, cw, fs)["err_rms_before"]
            for _ in range(2):
                apply_dtc_correction(pm, fit_dtc_correction(pm, cw, fs))
            after = fit_dtc_correction(pm, cw, fs)["err_rms_before"]
            return before, after

        self.run_async(job, lambda t: self.r3.setText(
            f"phase-error rms {t[0] * 1e3:.1f} → {t[1] * 1e3:.2f} mrad"),
            self.b3)


class MonteCarloPage(Page):
    title = "Monte Carlo"

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        box = QGroupBox("population")
        form = QFormLayout(box)
        self.n = QSpinBox()
        self.n.setRange(10, 200)
        self.n.setValue(40)
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.0, 1.5)
        self.sigma.setValue(0.5)
        self.sigma.setSingleStep(0.1)
        self.limit = QSpinBox()
        self.limit.setRange(-45, -25)
        self.limit.setValue(-35)
        self.cal = QCheckBox("per-chip skew calibration")
        form.addRow("chips", self.n)
        form.addRow("skew sigma [ns]", self.sigma)
        form.addRow("EVM limit [dB]", self.limit)
        form.addRow("", self.cal)
        self.btn = QPushButton("Run population")
        self.btn.clicked.connect(self._go)
        top = QHBoxLayout()
        top.addWidget(box, 1)
        top.addWidget(self.btn)
        lay.addLayout(top)
        self.summary = QLabel("—")
        lay.addWidget(self.summary)
        self.figbox = FigureBox()
        lay.addWidget(self.figbox, 1)

    def _go(self):
        n, sig = self.n.value(), self.sigma.value()
        lim, cal = float(self.limit.value()), self.cal.isChecked()
        self.run_async(
            lambda: run_mc_report(n, skew_sigma_ns=sig, calibrated=cal,
                                  limit_db=lim),
            self._show, self.btn)

    def _show(self, rep):
        s = rep["summary"]
        self.summary.setText(
            f"yield {100 * s['yield']:.0f}%  |  mean {s['mean']:.1f} dB, "
            f"std {s['std']:.1f}, worst {s['worst']:.1f}, "
            f"best {s['best']:.1f} dB")
        self.figbox.set_figure(rep["fig"])
