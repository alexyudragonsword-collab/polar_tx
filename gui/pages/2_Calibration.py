import streamlit as st

from polartx.guiutil import run_dtc_cal, run_skew_cal, run_twopoint_cal

st.title("Calibration lab")

tab1, tab2, tab3 = st.tabs(["AM/PM skew", "two-point gain", "DTC LUT"])

with tab1:
    skew_ns = st.slider("injected skew [ns]", 0.0, 4.0, 2.0, 0.1)
    method = st.radio("estimator", ["waveform align", "ACP power search"])
    if st.button("estimate skew"):
        r = run_skew_cal(skew_ns, method=method)
        st.metric("estimated skew", f"{r['estimated_ns']:.2f} ns",
                  f"error {r['error_ps']:.0f} ps")

with tab2:
    eps = st.slider("injected direct-path gain error [%]", -8.0, 8.0, 3.0)
    if st.button("estimate + correct"):
        r = run_twopoint_cal(eps)
        c1, c2, c3 = st.columns(3)
        c1.metric("estimated eps", f"{r['estimated_pct']:.2f} %")
        c2.metric("EVM before", f"{r['evm_before_db']:.1f} dB")
        c3.metric("EVM after", f"{r['evm_after_db']:.1f} dB")

with tab3:
    if st.button("run DTC gain/INL cal (CW training, 2 iterations)"):
        r = run_dtc_cal(2)
        st.metric("phase-error rms", f"{r['err_rms_after_mrad']:.2f} mrad",
                  f"was {r['err_rms_before_mrad']:.1f} mrad")
