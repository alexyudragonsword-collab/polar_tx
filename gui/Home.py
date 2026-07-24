"""polartx Streamlit workbench — run: streamlit run gui/Home.py"""
import streamlit as st

st.set_page_config(page_title="polartx", page_icon="📡", layout="wide")

st.title("polartx — digital polar TX workbench")
st.markdown("""
两种数字极坐标发射机架构、一条可组合链路：

- **窄带**：ADPLL 两点调制（BLE GFSK、BT EDR DPSK、LTE ≤20 MHz）
- **宽带**：开环 DTC 相位调制器（WiFi 6/7 ≤320 MHz、5G NR ≤200 MHz）

左侧页面：**Chain Workbench**（预设链路 + 损伤旋钮 + EVM/ACLR/mask），
**Calibration**（skew / 两点增益 / DTC LUT 校准演示），
**Monte Carlo**（失配良率分析）。
""")

st.info("所有计算都在 `polartx.guiutil`（纯 Python，可脱离 GUI 测试）；"
        "页面只做布局。命令行等价物见 examples/ex01–ex07。")
