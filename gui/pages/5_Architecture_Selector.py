"""Architecture selector: rank the four topologies against a requirement."""
import streamlit as st

from polartx.guiutil import run_selector_report

st.title("Architecture selector")

st.caption(
    "Four topologies, one shared synthesizer, so in-band phase noise is a "
    "common term. The ADPLL imprints modulation inside the loop (no DTC "
    "floors) but its direct FM DAC cannot stay gain-matched past ~50 MHz; "
    "the open-loop DTC works at any bandwidth and pays quantization / "
    "jitter / INL floors; outphasing inherits the DTC floors on two "
    "full-scale branches plus branch phase mismatch; the Cartesian RF-DAC "
    "pays I/Q quantization, LO jitter and image rejection. Scores are "
    "analytic — confirm the winner on the real chain.")

c1, c2, c3 = st.columns(3)
bw = c1.slider("signal bandwidth [MHz]", 1, 320, 80)
evm_target = c2.slider("EVM target [dB]", -45, -20, -35)
fout = c3.selectbox("carrier [GHz]", [2.44, 3.5, 5.8, 6.0], index=2)

c4, c5, c6 = st.columns(3)
dtc_bits = c4.slider("DTC resolution [bits]", 8, 14, 11)
match_pct = c5.slider("two-point gain match [%]", 0.1, 2.0, 0.2, 0.1,
                      help="0.2% = calibrated (online sign-sign LMS); "
                           "0.5% = uncalibrated")
mod = c6.selectbox("modulation", ["ofdm", "gfsk", "dpsk", "qam"], index=0)

const_env = st.checkbox("constant envelope (GFSK/GMSK)", False)

if st.button("Rank architectures", type="primary"):
    rep = run_selector_report(bw_hz=bw * 1e6, evm_db_max=float(evm_target),
                              fout=fout * 1e9, dtc_bits=dtc_bits,
                              modulation=mod,
                              two_point_gain_match=match_pct / 100.0,
                              constant_envelope=const_env)
    m = rep["metrics"]
    st.subheader(m["recommendation"])
    st.table({k: v for k, v in m.items() if k != "recommendation"})
    st.code(rep["table"], language=None)
    st.pyplot(rep["fig"])
