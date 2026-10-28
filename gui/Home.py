"""polartx Streamlit workbench — run: streamlit run gui/Home.py"""
import streamlit as st

st.set_page_config(page_title="polartx", page_icon="📡", layout="wide")

st.title("polartx — digital polar TX workbench")
st.markdown("""
数字发射机行为级仿真，四种拓扑共用一条可组合链路：

- **窄带极坐标**：ADPLL 两点调制（BLE GFSK、BT EDR DPSK、LTE ≤20 MHz）
- **宽带极坐标**：开环 DTC 相位调制器（WiFi 6/7 ≤320 MHz、5G NR ≤200 MHz）
- **Outphasing**：两路恒包络 DTC 分支 + Chireix 合路
- **笛卡尔 RF-DAC**：I/Q 两组电流单元阵列，无包络/相位通路

左侧页面：

- **Chain Workbench** — 预设链路 + 该预设接受的损伤旋钮 + EVM / 带内发射或 ACLR / mask
- **Calibration** — skew 估计、两点增益、DTC 增益/INL LUT 校准
- **Monte Carlo** — 失配良率
- **FIR Doherty** — Borokhovich RFIC'26 双抽头 FIR 陷波 + 多核 Doherty 合路
- **Architecture Selector** — 四种拓扑按需求排序推荐
- **RTL Export** — 数字数据通路 + Verilog-AMS 实数 PA 模型，iverilog 逐位校验
""")

st.info("所有计算都在 `polartx.guiutil`（纯 Python，可脱离 GUI 测试）；"
        "页面只做布局。命令行等价物见 `examples/` 下的脚本。")
