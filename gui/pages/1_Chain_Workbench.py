import streamlit as st

from polartx.guiutil import (DEFAULT_PRESET, PRESETS, chain_overrides,
                             honoured_knobs, run_chain_report)

st.title("Chain Workbench")

col = st.sidebar
name = col.selectbox("preset", PRESETS, index=PRESETS.index(DEFAULT_PRESET))
seed = col.number_input("seed", 0, 9999, 1)
noise = col.checkbox("noise on", True)

# guiutil decides which knobs reach which chain; this page shows exactly
# those, so a knob that does nothing on this preset is never offered
keys = honoured_knobs(name)
values: dict = {}
cfr_on = True
if not keys:
    col.caption("literature-class benchmark — parameters fixed to the "
                "published-class assumptions (see the preset docstring)")
if "n_bits" in keys:
    values["n_bits"] = col.slider("DTC bits", 6, 14, 11)
if "cfr_papr_db" in keys:
    cfr_on = col.checkbox("CFR 8.5 dB", True)
if "dpd" in keys:
    values["dpd"] = col.checkbox("polar DPD", True)
if "dp_gain" in keys:
    values["dp_gain"] = 1.0 + col.slider("two-point gain error [%]",
                                         -10.0, 10.0, 0.0, 0.5) / 100
if "loop_bw" in keys:
    values["loop_bw"] = col.slider(
        "loop BW [kHz]", 50, 400, 100,
        help="with the two points matched this shapes only the noise; "
             "set a gain error to see it trade against mismatch") * 1e3
if "env_skew_s" in keys:
    # skew is an ENVELOPE-path impairment: quasi-constant-envelope EDR
    # tolerates tens of ns (10 ns ~ 0.13% DEVM), OFDM fails its mask at
    # 1 ns (LTE-20) or 0.2 ns (WiFi 160 MHz)
    span = 20.0 if "EDR" in name else 4.0
    values["env_skew_s"] = col.slider("AM/PM skew [ns]", -span, span, 0.0,
                                      span / 40) * 1e-9
overrides = chain_overrides(name, cfr_off=not cfr_on, **values)

if st.button("Run", type="primary"):
    with st.spinner("simulating..."):
        rep = run_chain_report(name, seed=int(seed), noise=noise,
                               **overrides)
    # stringify: the metrics dict mixes floats with the "PASS"/"FAIL" mask
    # verdict, and Arrow rejects that mixed column (Streamlit recovers, but
    # logs a serialization traceback each run)
    st.table({k: str(v) for k, v in rep["metrics"].items()})
    st.pyplot(rep["fig"])
