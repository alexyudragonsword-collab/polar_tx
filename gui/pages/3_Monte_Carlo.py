import streamlit as st

from polartx.guiutil import run_mc_report

st.title("Monte Carlo yield")

n = st.sidebar.slider("chips", 10, 200, 40, 10)
skew_ns = st.sidebar.slider("skew sigma [ns]", 0.0, 1.5, 0.5, 0.1)
limit = st.sidebar.slider("EVM limit [dB]", -45, -25, -35)
cal = st.sidebar.checkbox("per-chip skew calibration", False)

if st.button("Run population", type="primary"):
    with st.spinner(f"running {n} chips..."):
        rep = run_mc_report(n, skew_sigma_ns=skew_ns, calibrated=cal,
                            limit_db=float(limit))
    st.table({k: round(v, 2) if isinstance(v, float) else v
              for k, v in rep["summary"].items()})
    st.pyplot(rep["fig"])
