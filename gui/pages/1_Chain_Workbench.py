import streamlit as st

from polartx.guiutil import PRESETS, run_chain_report

st.title("Chain Workbench")

col = st.sidebar
name = col.selectbox("preset", PRESETS, index=6)
seed = col.number_input("seed", 0, 9999, 1)
noise = col.checkbox("noise on", True)

overrides = {}
if name.startswith("Bench:"):
    col.caption("literature-class benchmark — parameters fixed to the "
                "published-class assumptions (see the preset docstring)")
elif "WiFi" in name or "NR" in name:
    overrides["n_bits"] = col.slider("DTC bits", 6, 14, 11)
    overrides["env_skew_s"] = col.slider("AM/PM skew [ns]", -4.0, 4.0,
                                         0.0, 0.1) * 1e-9
    if col.checkbox("CFR 8.5 dB", True) is False:
        overrides["cfr_papr_db"] = None
elif "LTE" in name:
    overrides["dpd"] = col.checkbox("polar DPD", True)
    overrides["dp_gain"] = 1.0 + col.slider("two-point gain error [%]",
                                            -10.0, 10.0, 0.0, 0.5) / 100
    # skew is an ENVELOPE-path impairment: the narrowband OFDM chain is
    # ~8x more tolerant than 160 MHz WiFi (in proportion to bandwidth),
    # not immune — 1 ns already fails the mask.
    overrides["env_skew_s"] = col.slider("AM/PM skew [ns]", -4.0, 4.0,
                                         0.0, 0.1) * 1e-9
elif "BLE" in name or "EDR" in name:
    overrides["dp_gain"] = 1.0 + col.slider("two-point gain error [%]",
                                            -10.0, 10.0, 0.0, 0.5) / 100
    overrides["loop_bw"] = col.slider("loop BW [kHz]", 50, 400, 100) * 1e3
    overrides["env_skew_s"] = col.slider(
        "AM/PM skew [ns]", -20.0, 20.0, 0.0, 0.5,
        help="GFSK is constant-envelope and provably immune; EDR DPSK is "
             "quasi-constant, so 10 ns costs only ~0.13% DEVM") * 1e-9

if st.button("Run", type="primary"):
    with st.spinner("simulating..."):
        rep = run_chain_report(name, seed=int(seed), noise=noise,
                               **overrides)
    # stringify: the metrics dict mixes floats with the "PASS"/"FAIL" mask
    # verdict, and Arrow rejects that mixed column (Streamlit recovers, but
    # logs a serialization traceback each run)
    st.table({k: str(v) for k, v in rep["metrics"].items()})
    st.pyplot(rep["fig"])
