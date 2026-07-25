"""Feature pages, all computation via the headless polartx.guiutil layer."""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QSpinBox, QVBoxLayout)

from ..guiutil import (PRESETS, run_chain_report, run_combiner_report,
                       run_fir_report, run_mc_report, run_rtl_export,
                       run_selector_report)
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
        self.btn_save = QPushButton("Save setup…")
        self.btn_save.clicked.connect(self._save)
        self.btn_load = QPushButton("Load setup…")
        self.btn_load.clicked.connect(self._load)
        side = QVBoxLayout()
        side.addWidget(self.btn)
        side.addWidget(self.btn_save)
        side.addWidget(self.btn_load)
        side.addStretch(1)
        top = QHBoxLayout()
        top.addWidget(form_box, 1)
        top.addLayout(side)
        lay.addLayout(top)
        self.table_slot = QVBoxLayout()
        lay.addLayout(self.table_slot)
        self.figbox = FigureBox()
        lay.addWidget(self.figbox, 1)
        self._table = None

    def _overrides(self, name: str) -> dict:
        ov = {}
        if name.startswith("Bench:"):
            return ov            # benchmarks fix their published-class params
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

    # -------------------------------------------------- setup save/load
    def _save(self):
        from PySide6.QtWidgets import QFileDialog

        from ..guiutil import save_setup
        path, _ = QFileDialog.getSaveFileName(self, "Save setup", "",
                                              "polartx setup (*.json)")
        if path:
            name = self.preset.currentText()
            save_setup(path, name, self._overrides(name),
                       seed=self.seed.value(),
                       noise=self.noise.isChecked())

    def apply_setup(self, doc: dict):
        """Push a loaded setup back into the widgets."""
        self.preset.setCurrentText(doc["preset"])
        self.seed.setValue(int(doc["seed"]))
        self.noise.setChecked(bool(doc["noise"]))
        ov = doc["overrides"]
        if "n_bits" in ov:
            self.dtc_bits.setValue(int(ov["n_bits"]))
        if "env_skew_s" in ov:
            self.skew_ns.setValue(1e9 * float(ov["env_skew_s"]))
        if "dp_gain" in ov:
            self.dp_err_pct.setValue(100.0 * (float(ov["dp_gain"]) - 1.0))
        if "dpd" in ov:
            self.dpd.setChecked(bool(ov["dpd"]))

    def _load(self):
        from PySide6.QtWidgets import QFileDialog

        from ..guiutil import load_setup
        path, _ = QFileDialog.getOpenFileName(self, "Load setup", "",
                                              "polartx setup (*.json)")
        if path:
            self.apply_setup(load_setup(path))


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


class FIRDohertyPage(Page):
    """Borokhovich RFIC'26: 2-tap FIR notch and multi-core Doherty
    combining — the two halves of that paper, side by side."""

    title = "FIR + Doherty"

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)

        fir_box = QGroupBox("2-tap FIR notch (dual DPA chains)")
        ff = QFormLayout(fir_box)
        self.fir_bw = QComboBox()
        self.fir_bw.addItems(["40", "80", "160"])
        self.notch = QSpinBox()
        self.notch.setRange(100, 900)
        self.notch.setValue(500)
        self.notch.setSingleStep(50)
        self.fir_syms = QSpinBox()
        self.fir_syms.setRange(4, 48)
        self.fir_syms.setValue(16)
        ff.addRow("bandwidth [MHz]", self.fir_bw)
        ff.addRow("notch offset [MHz]", self.notch)
        ff.addRow("OFDM symbols", self.fir_syms)
        self.fir_btn = QPushButton("Run FIR chain")
        self.fir_btn.clicked.connect(self._go_fir)
        ff.addRow("", self.fir_btn)

        doh_box = QGroupBox("Doherty combiner (derived from load modulation)")
        df = QFormLayout(doh_box)
        self.n_way = QComboBox()
        self.n_way.addItems(["2", "3"])
        self.backoff = QDoubleSpinBox()
        self.backoff.setRange(3.0, 12.0)
        self.backoff.setValue(6.0)
        self.backoff.setSingleStep(0.5)
        self.peaking = QComboBox()
        self.peaking.addItems(["C", "B"])
        self.gain_imb = QDoubleSpinBox()
        self.gain_imb.setRange(0.0, 20.0)
        self.gain_imb.setSingleStep(1.0)
        self.ph_imb = QDoubleSpinBox()
        self.ph_imb.setRange(0.0, 20.0)
        self.ph_imb.setSingleStep(1.0)
        df.addRow("cores (n-way)", self.n_way)
        df.addRow("backoff point [dB]", self.backoff)
        df.addRow("peaking class", self.peaking)
        df.addRow("gain imbalance [%]", self.gain_imb)
        df.addRow("phase imbalance [deg]", self.ph_imb)
        self.doh_btn = QPushButton("Run combiner")
        self.doh_btn.clicked.connect(self._go_doherty)
        df.addRow("", self.doh_btn)

        top = QHBoxLayout()
        top.addWidget(fir_box, 1)
        top.addWidget(doh_box, 1)
        lay.addLayout(top)
        self.table_slot = QVBoxLayout()
        lay.addLayout(self.table_slot)
        self._table = None
        self.figbox = FigureBox()
        lay.addWidget(self.figbox, 1)

    def _go_fir(self):
        bw = float(self.fir_bw.currentText()) * 1e6
        off = float(self.notch.value()) * 1e6
        n = self.fir_syms.value()
        self.run_async(
            lambda: run_fir_report(bw=bw, notch_offset_hz=off, n_symbols=n),
            self._show, self.fir_btn, self.doh_btn)

    def _go_doherty(self):
        n = int(self.n_way.currentText())
        bo = self.backoff.value()
        pk = self.peaking.currentText()
        gi, pi_ = self.gain_imb.value(), self.ph_imb.value()
        self.run_async(
            lambda: run_combiner_report(n_way=n, backoff_db=bo, peaking=pk,
                                        gain_imbalance_pct=gi,
                                        phase_imbalance_deg=pi_),
            self._show, self.fir_btn, self.doh_btn)

    def _show(self, rep):
        if self._table is not None:
            self.table_slot.removeWidget(self._table)
            self._table.deleteLater()
        self._table = metrics_table(rep["metrics"])
        self.table_slot.addWidget(self._table)
        self.figbox.set_figure(rep["fig"])


class SelectorPage(Page):
    """Rank narrowband ADPLL two-point vs wideband open-loop DTC."""

    title = "Architecture Selector"

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        box = QGroupBox("requirement")
        form = QFormLayout(box)
        self.bw = QSpinBox()
        self.bw.setRange(1, 320)
        self.bw.setValue(80)
        self.evm = QSpinBox()
        self.evm.setRange(-45, -20)
        self.evm.setValue(-35)
        self.fout = QComboBox()
        self.fout.addItems(["2.44", "3.5", "5.8", "6.0"])
        self.fout.setCurrentIndex(2)
        self.dtc_bits = QSpinBox()
        self.dtc_bits.setRange(8, 14)
        self.dtc_bits.setValue(11)
        self.match = QDoubleSpinBox()
        self.match.setRange(0.1, 2.0)
        self.match.setValue(0.2)
        self.match.setSingleStep(0.1)
        self.mod = QComboBox()
        self.mod.addItems(["ofdm", "gfsk", "dpsk", "qam"])
        self.const_env = QCheckBox("constant envelope")
        form.addRow("bandwidth [MHz]", self.bw)
        form.addRow("EVM target [dB]", self.evm)
        form.addRow("carrier [GHz]", self.fout)
        form.addRow("DTC bits", self.dtc_bits)
        form.addRow("two-point match [%]", self.match)
        form.addRow("modulation", self.mod)
        form.addRow("", self.const_env)
        self.btn = QPushButton("Rank architectures")
        self.btn.clicked.connect(self._go)
        top = QHBoxLayout()
        top.addWidget(box, 1)
        top.addWidget(self.btn)
        lay.addLayout(top)
        self.rec = QLabel("—")
        self.rec.setWordWrap(True)
        lay.addWidget(self.rec)
        self.figbox = FigureBox()
        lay.addWidget(self.figbox, 1)

    def _go(self):
        bw = float(self.bw.value()) * 1e6
        evm = float(self.evm.value())
        fo = float(self.fout.currentText()) * 1e9
        bits = self.dtc_bits.value()
        match = self.match.value() / 100.0
        mod = self.mod.currentText()
        ce = self.const_env.isChecked()
        self.run_async(
            lambda: run_selector_report(bw_hz=bw, evm_db_max=evm, fout=fo,
                                        dtc_bits=bits, modulation=mod,
                                        two_point_gain_match=match,
                                        constant_envelope=ce),
            self._show, self.btn)

    def _show(self, rep):
        m = rep["metrics"]
        self.rec.setText(f"{m['recommendation']}\n\nclosest preset: "
                         f"{m['closest preset']}")
        self.figbox.set_figure(rep["fig"])


class RTLPage(Page):
    """Emit the digital datapath + Verilog-AMS PA model, verify bit-true."""

    title = "RTL / AMS Export"

    def __init__(self):
        super().__init__()
        import os
        import tempfile
        lay = QVBoxLayout(self)
        box = QGroupBox("datapath")
        form = QFormLayout(box)
        self.n_bits = QSpinBox()
        self.n_bits.setRange(6, 12)
        self.n_bits.setValue(10)
        self.n_thermo = QSpinBox()
        self.n_thermo.setRange(0, 12)
        self.n_thermo.setValue(7)
        self.with_dpd = QCheckBox("include polar-DPD dual LUT")
        self.with_dpd.setChecked(True)
        self.verify = QCheckBox("run iverilog golden checks")
        self.verify.setChecked(True)
        form.addRow("DPA bits", self.n_bits)
        form.addRow("thermometer MSBs", self.n_thermo)
        form.addRow("", self.with_dpd)
        form.addRow("", self.verify)
        self._outdir = os.path.join(tempfile.gettempdir(), "polartx_rtl")
        form.addRow("output dir", QLabel(self._outdir))
        self.btn = QPushButton("Emit RTL")
        self.btn.clicked.connect(self._go)
        top = QHBoxLayout()
        top.addWidget(box, 1)
        top.addWidget(self.btn)
        lay.addLayout(top)
        self.table_slot = QVBoxLayout()
        lay.addLayout(self.table_slot)
        self._table = None
        self.files = QLabel("—")
        self.files.setWordWrap(True)
        lay.addWidget(self.files, 1)

    def _go(self):
        nb, nt = self.n_bits.value(), self.n_thermo.value()
        dpd, ver = self.with_dpd.isChecked(), self.verify.isChecked()
        self.run_async(
            lambda: run_rtl_export(self._outdir, n_bits=nb,
                                   n_thermo=min(nt, nb), with_dpd=dpd,
                                   verify=ver),
            self._show, self.btn)

    def _show(self, rep):
        if self._table is not None:
            self.table_slot.removeWidget(self._table)
            self._table.deleteLater()
        self._table = metrics_table(rep["checks"])
        self.table_slot.addWidget(self._table)
        self.files.setText(f"{len(rep['files'])} files in {rep['outdir']}:\n"
                           + ", ".join(rep["files"]))
