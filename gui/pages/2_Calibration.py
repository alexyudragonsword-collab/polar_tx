import numpy as np
import streamlit as st

st.title("Calibration lab")

tab1, tab2, tab3 = st.tabs(["AM/PM skew", "two-point gain", "DTC LUT"])

with tab1:
    skew_ns = st.slider("injected skew [ns]", 0.0, 4.0, 2.0, 0.1)
    method = st.radio("estimator", ["waveform align", "ACP power search"])
    if st.button("estimate skew"):
        from polartx.cal.skew import estimate_env_skew, estimate_skew_by_acp
        from polartx.presets import wifi_dtc
        p = wifi_dtc(bw=160e6, env_skew_s=skew_ns * 1e-9)
        wf = p.make_waveform(n_symbols=4, seed=2)
        if method == "waveform align":
            est = estimate_env_skew(p.tx.run(wf, noise=False))
        else:
            est = estimate_skew_by_acp(p.tx, wf)
        st.metric("estimated skew", f"{est['skew_s'] * 1e9:.2f} ns",
                  f"error {abs(est['skew_s'] - skew_ns * 1e-9) * 1e12:.0f} ps")

with tab2:
    eps = st.slider("injected direct-path gain error [%]", -8.0, 8.0, 3.0)
    if st.button("estimate + correct"):
        from polartx.cal.twopoint import estimate_dp_gain_error
        from polartx.presets import lte20_adpll
        p = lte20_adpll(qam=64, dp_gain=1 + eps / 100)
        wf = p.make_waveform(n_symbols=10, seed=0)
        r0 = p.tx.run(wf, noise=True, seed=2)
        est = estimate_dp_gain_error(p.tx.phasemod, r0.phase_cmd,
                                     r0.phase_out, r0.fs)
        p.tx.phasemod.dp_gain *= est["dp_gain_corr"]
        r1 = p.tx.run(wf, noise=True, seed=2)
        c1, c2, c3 = st.columns(3)
        c1.metric("estimated eps", f"{100 * est['eps_hat']:.2f} %")
        c2.metric("EVM before", f"{r0.evm().db:.1f} dB")
        c3.metric("EVM after", f"{r1.evm().db:.1f} dB")

with tab3:
    if st.button("run DTC gain/INL cal (CW training, 2 iterations)"):
        from polartx.cal.dtc_cal import (apply_dtc_correction,
                                         fit_dtc_correction)
        from polartx.phasemod import DTCPhaseModulator, DTCPMConfig
        FS, N = 640e6, 1 << 16
        cw = 2 * np.pi * 5e6 * np.arange(N) / FS
        pm = DTCPhaseModulator(DTCPMConfig(
            n_bits=12, gain_error=0.01, inl_sin=(2e-3, 3, 0.0)))
        before = fit_dtc_correction(pm, cw, FS)["err_rms_before"]
        for _ in range(2):
            apply_dtc_correction(pm, fit_dtc_correction(pm, cw, FS))
        after = fit_dtc_correction(pm, cw, FS)["err_rms_before"]
        st.metric("phase-error rms", f"{after * 1e3:.2f} mrad",
                  f"was {before * 1e3:.1f} mrad")
