"""RTL / Verilog-AMS export of the digital polar-TX datapath."""
import os
import tempfile

import streamlit as st

from polartx.guiutil import run_rtl_export

st.title("RTL / AMS export")

st.caption(
    "Emits the synthesizable datapath — CFR clip, DTC phase accumulator, "
    "DPA segmented thermometer decoder, polar-DPD dual LUT — plus a "
    "Verilog-AMS `wreal` real-number model of the DPA (the digital/analog "
    "co-sim bridge). Every digital block is checked bit-true against its "
    "Python golden with iverilog when it is installed.")

c1, c2 = st.columns(2)
n_bits = c1.slider("DPA resolution [bits]", 6, 12, 10)
n_thermo = c2.slider("thermometer MSBs", 0, n_bits, min(7, n_bits))
with_dpd = st.checkbox("include polar-DPD dual LUT", True)
verify = st.checkbox("run iverilog golden checks", True)

outdir = st.text_input("output directory",
                       os.path.join(tempfile.gettempdir(), "polartx_rtl"))

if st.button("Emit RTL", type="primary"):
    with st.spinner("emitting + verifying..."):
        rep = run_rtl_export(outdir, n_bits=n_bits, n_thermo=n_thermo,
                             with_dpd=with_dpd, verify=verify)
    st.success(f"{len(rep['files'])} files written to {rep['outdir']}")
    if rep["checks"]:
        st.subheader("bit-true verification")
        st.table(rep["checks"])
    st.subheader("files")
    st.code("\n".join(rep["files"]), language=None)
