# polartx — 数字极坐标发射机（Digital Polar TX）行为级仿真

面向 RFIC 系统设计的数字极坐标发射机 Python 行为仿真库，覆盖两种架构、一条可组合的信号链：

```
Waveform → [CFR] → polar split → 包络路径（量化/skew/DPA 幅度码）─┐
                              └ 相位路径（PhaseModulator）────────┤→ DPA 重构 → EVM/ACLR/Mask/PSD
```

| 架构 | 相位调制器 | 应用 | 带宽 |
|---|---|---|---|
| **窄带** | ADPLL 两点调制（`ADPLLTwoPoint`，response/event 双引擎交叉验证） | 蓝牙 BLE GFSK、4G LTE | ≤ 20 MHz |
| **宽带** | 开环 DTC 相位调制器（`DTCPhaseModulator`，逐样本向量化） | WiFi 6/7、5G NR | ≤ 320 MHz |

相位路径引擎改编自姊妹库 [`pll_simulator`](../pll_simulator)（ADPLL/DTC/DCO/相噪/ΣΔ/LMS），波形与指标基础设施改编自 [`PA_DPD`](../PA_DPD)（OFDM/QAM/EVM/ACLR/Mask/CCDF/CFR/延迟对齐）——见 `src/polartx/vendor/`（逐文件注明出处，单一接缝，可随时切回 pip 依赖）。

## 快速开始

```bash
pip install -e .          # numpy / scipy / matplotlib
pytest tests/             # 36 项测试：物理量断言 + 与 padpd 逐位回归
python examples/ex01_ble_gfsk_adpll.py      # 图落在 examples/out/
```

```python
from polartx import ble_1m_adpll, wifi_dtc

p = ble_1m_adpll()                          # BLE LE-1M @ 2.44 GHz, fref 32 MHz
res = p.tx.run(p.make_waveform(600), seed=1)
print(res.evm(), res.check_mask()[0])       # 相位轨迹 EVM、BLE 发射 mask

p = wifi_dtc(bw=320e6, qam=4096)            # WiFi 7 @ 1.28 GS/s 基带
res = p.tx.run(p.make_waveform(), seed=1)
print(res.evm().db, res.aclr())             # -39.6 dB, ACLR ~ -58 dBc
```

## 首期结果（预设默认值）

| 链路 | 配置 | EVM | ACLR / Mask |
|---|---|---|---|
| BLE LE-1M | ADPLL 两点，环路 BW 100 kHz，−112 dBc/Hz DCO | 2.7%（相噪限制） | mask PASS |
| BLE，直通点增益误差 5% | 同上（两点校准规格图见 ex01） | 6.4% | — |
| BT EDR2 π/4-DQPSK / EDR3 8DPSK | 同一 ADPLL 相位路径 + **非恒包络** SRRC 包络路径（PAPR ~3.2 dB） | DEVM 1.8% / 1.8%（限值 20/13%） | mask PASS |
| WiFi 6 80/160 MHz 1024-QAM | 11-bit dithered DTC + 10-bit DPA + CFR 8.5 dB | **−40.9 / −39.5 dB** | −54/−58 dBc，PASS |
| WiFi 7 320 MHz 4096-QAM | 同上，WiFi-7 级锁定 LO | **−39.6 dB**（−38 达标） | −58 dBc，PASS |

关键设计数据（examples 复现）：

- **两点调制**：response（线性化 z 域，整帧秒级）与 event（逐参考周期引擎）两引擎在失配全扫描上 EVM 吻合 <0.3%；匹配时 EVM 与环路带宽无关（测试固化）。
- **DTC 分辨率设计图**（ex03）：相位量化 EVM 与解析式 `(2π/2^B)/√12·√(1/osr)` 差 <0.2 dB；4096-QAM 需 ~10 bit，一阶误差反馈 dither 省 ~1 bit；INL 正弦项杂散、ZOH 更新时钟镜像均与闭式预测吻合（1–2 dB 内，测试断言）。
- **AM/PM skew**（ex04）：160 MHz 下 0.5 ns skew 即把 EVM 从 −39.5 打到 −21.8 dB；互谱相位斜率估计精度 0.05 采样，校正后完全恢复——极坐标架构最尖锐的损伤，就是这条链的核心卖点。
- **带宽扩展**（ex02）：80 MHz OFDM 极坐标分解后包络/相位路径 99% 占用带宽 2.1×/3.2×；幅度域 hole punching 的 EVM 代价闭式可算（测试断言到 0.01 dB）。

## 包结构

```
src/polartx/
├── vendor/            # pllsim / padpd 改编移植子集（出处注释 + 单一接缝）
├── waveforms/         # Waveform 容器；BLE GFSK；BT EDR π/4-DQPSK/8DPSK（SRRC）；通用 OFDM（SCS 可配，WiFi 预设与 padpd 逐位一致）
├── polar/             # 极坐标分解/重构、hole punching、带宽扩展分析
├── phasemod/          # PhaseModulator ABC；ADPLLTwoPoint；DTCPhaseModulator
├── dpa/               # 温度计+二进制单元阵列失配（√N 律）、AM-AM(Rapp/LUT)/AM-PM、码表 DPA
├── chain.py           # ChainConfig + PolarTX + PolarResult（evm/aclr/psd/check_mask）
├── impairments.py     # 分数延迟 skew、ZOH
├── cal/               # AM/PM skew 估计（互谱相位斜率）与校正
├── metrics/           # EVM/ACLR/Mask/CCDF/AM-AM（vendored）+ BLE Δf1/Δf2、EDR DEVM、BLE/BT mask
├── analysis/          # 解析对照：量化噪底、INL 杂散、ZOH 镜像、包络量化 EVM
└── presets.py         # ble_1m/2m_adpll、wifi_dtc(80/160/320)
```

## 建模口径与已知边界

- 波形为**风格化**模型：正确 numerology（SCS/FFT/CP/占用带宽）+ 随机 QAM 数据音，无导频/前导/信道编码；mask 为工程化模板，非认证测试器（沿用 padpd 措辞）。
- `ADPLLTwoPoint` response 模式是线性化模型（无 TDC 回绕、dither×调制耦合）；杂散类结论以 event 模式与解析预测背书。event 模式逐参考周期（Python 循环 ~1–2 Mcycles/s），长帧验证用短段。
- 幅度域 hole punching 钳制包络动态范围与包络带宽，但相位路径保留 π 翻转（DTC 按 mod 2π 处理）；相位轨迹平滑是后续里程碑。
- LO 相噪用"环内平坦"锁定近似（`lo_loop_bw`）。

## 路线图

- **M2 — LTE 20 MHz 窄带 polar 全链路**：LTE numerology（fft_size 与信道带宽解耦）、E-UTRA ACLR/SEM、RX 频段噪声解析预算（NoisePath 合成）、polar DPD（AM-AM/AM-PM 逆 LUT，ILA 拟合）、两点增益 LMS 在线校准。
- **M3 — 5G NR + 高级校准**：NR SCS 30/120 kHz @ 100/200 MHz、NR SEM/ACLR 口径、开环 DTC 增益/INL LUT 校准（复用 pllsim LUTCal）、DPA 交织与镜像研究、谱不对称 skew 估计、GMP 记忆效应包装。
- **暂缓**：GUI（Streamlit/Qt，姊妹库模式可直接照搬）、Monte Carlo 良率、RTL/AMS 导出。

## Examples

| 脚本 | 内容 |
|---|---|
| `ex01_ble_gfsk_adpll.py` | BLE 频率轨迹/眼图、EVM vs 两点失配设计图（双引擎）、Δf1/Δf2、BLE mask、相噪分解 |
| `ex02_polar_bandwidth_expansion.py` | 极坐标带宽扩展、包络零点、hole-punch 权衡 |
| `ex03_dtc_phase_modulator.py` | DTC 分辨率设计图（含 dither）、INL 杂散 vs 预测、ZOH 镜像、LO 相噪贡献 |
| `ex04_wifi_polar_chain.py` | WiFi 80/160/320 EVM/ACLR/Mask 总表、星座图、CFR CCDF、skew 灵敏度与校准闭环 |
