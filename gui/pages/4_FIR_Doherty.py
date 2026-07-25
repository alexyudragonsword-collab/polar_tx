"""2-tap FIR notch + multi-core Doherty combining (Borokhovich RFIC'26)."""
import streamlit as st

from polartx.guiutil import run_combiner_report, run_fir_report

st.title("FIR notch + Doherty combining")

tab_fir, tab_doherty = st.tabs(["2-tap FIR benchmark", "Doherty combiner"])

with tab_fir:
    st.caption(
        "Borokhovich, Socher & Degani, RFIC 2026 — two DPA chains combined "
        "with a tap delay tau = 1/(2*offset), giving H = 1 + exp(-j2*pi*f*tau). "
        "The notch suppresses **correlated** out-of-channel content only, so "
        "the realistic depth is bounded by the random noise floor.")
    c1, c2, c3 = st.columns(3)
    bw = c1.selectbox("bandwidth [MHz]", [40, 80, 160], index=0)
    notch = c2.slider("notch offset [MHz]", 100, 900, 500, 50)
    nsym = c3.slider("OFDM symbols", 4, 48, 16,
                     help="EVM here is CFR-clipping-limited and bursty; "
                          "short records swing many dB seed to seed")
    noise = st.checkbox("noise on", True)

    if st.button("Run FIR chain", type="primary", key="fir"):
        with st.spinner("running dual-tap + single-tap baseline..."):
            rep = run_fir_report(bw=bw * 1e6, notch_offset_hz=notch * 1e6,
                                 n_symbols=nsym, noise=noise)
        st.table(rep["metrics"])
        st.pyplot(rep["fig"])

with tab_doherty:
    st.caption(
        "Power combining modeled from the load modulation, not a fitted "
        "curve: the main core saturates at the backoff point, then the "
        "peaking core(s) ramp to full power. Core gain/phase imbalance "
        "shows up as the AM-AM/AM-PM handoff kink.")
    c1, c2, c3 = st.columns(3)
    n_way = c1.selectbox("cores (n-way)", [2, 3], index=0)
    backoff = c2.slider("backoff point [dB]", 3.0, 12.0, 6.0, 0.5)
    peaking = c3.selectbox("peaking class", ["C", "B"], index=0,
                           help="B = ideal flat top; C = realistic dip")
    c4, c5, c6 = st.columns(3)
    loss = c4.slider("combiner insertion loss [dB]", 0.0, 1.5, 0.4, 0.1)
    gain_imb = c5.slider("core gain imbalance [%]", 0.0, 20.0, 0.0, 1.0)
    ph_imb = c6.slider("core phase imbalance [deg]", 0.0, 20.0, 0.0, 1.0)

    if st.button("Run combiner", type="primary", key="doherty"):
        rep = run_combiner_report(n_way=n_way, backoff_db=backoff,
                                  peaking=peaking, combiner_loss_db=loss,
                                  gain_imbalance_pct=gain_imb,
                                  phase_imbalance_deg=ph_imb)
        st.table(rep["metrics"])
        st.pyplot(rep["fig"])
