import streamlit as st

from polartx.guiutil import PRESETS, run_chain_report

st.title("Chain Workbench")

col = st.sidebar
name = col.selectbox("preset", PRESETS, index=6)
seed = col.number_input("seed", 0, 9999, 1)
noise = col.checkbox("noise on", True)

overrides = {}
if "WiFi" in name or "NR" in name:
    overrides["n_bits"] = col.slider("DTC bits", 6, 14, 11)
    overrides["env_skew_s"] = col.slider("AM/PM skew [ns]", -4.0, 4.0,
                                         0.0, 0.1) * 1e-9
    if col.checkbox("CFR 8.5 dB", True) is False:
        overrides["cfr_papr_db"] = None
elif "LTE" in name:
    overrides["dpd"] = col.checkbox("polar DPD", True)
    overrides["dp_gain"] = 1.0 + col.slider("two-point gain error [%]",
                                            -10.0, 10.0, 0.0, 0.5) / 100
elif "BLE" in name or "EDR" in name:
    overrides["dp_gain"] = 1.0 + col.slider("two-point gain error [%]",
                                            -10.0, 10.0, 0.0, 0.5) / 100
    overrides["loop_bw"] = col.slider("loop BW [kHz]", 50, 400, 100) * 1e3

if st.button("Run", type="primary"):
    with st.spinner("simulating..."):
        rep = run_chain_report(name, seed=int(seed), noise=noise,
                               **overrides)
    st.table(rep["metrics"])
    st.pyplot(rep["fig"])
