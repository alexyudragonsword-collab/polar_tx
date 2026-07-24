# polartx — 数字极坐标发射机（Digital Polar TX）行为级仿真

面向 RFIC 系统设计的数字极坐标发射机 Python 行为仿真库，覆盖两种架构、一条可组合的信号链。

**图文设计指南**（中英双语）：浏览器打开 [`docs/index.html`](docs/index.html)——设计原理、九个成套示例的图文解读、经验教训、复现方法。

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
pytest tests/             # 80+ 项测试：物理量断言 + 与 padpd 逐位回归
python examples/ex01_ble_gfsk_adpll.py      # 图落在 examples/out/

pip install -e .[gui]     # Streamlit 网页工作台
streamlit run gui/Home.py

pip install -e .[guiqt]   # PySide6 原生桌面版
polartx-gui               # 或 python -m polartx.guiqt
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
| LTE 20 MHz 64/256-QAM（M2） | ADPLL 两点 @122.88 MHz + 压缩性 DPA + polar DPD | **−53.3 dB**（无 DPD −37.2） | ACLR1 −62.6 dBc，SEM PASS |
| 5G NR FR1 100 MHz 256-QAM（M3） | 开环 DTC @3.5 GHz，30 kHz SCS，491.52 MS/s | **−37.1 dB**（−29 达标） | ACLR1 −61.4 dBc，OBUE PASS |
| 5G NR FR2 200 MHz 64-QAM（M3） | 开环 DTC @28 GHz，120 kHz SCS，983.04 MS/s，毫米波 LO 受限 | **−27.6 dB**（−22 达标） | ACLR1 −47.5 dBc，OBUE PASS |

关键设计数据（examples 复现）：

- **两点调制**：response（线性化 z 域，整帧秒级）与 event（逐参考周期引擎）两引擎在失配全扫描上 EVM 吻合 <0.3%；匹配时 EVM 与环路带宽无关（测试固化）。
- **DTC 分辨率设计图**（ex03）：相位量化 EVM 与解析式 `(2π/2^B)/√12·√(1/osr)` 差 <0.2 dB；4096-QAM 需 ~10 bit，一阶误差反馈 dither 省 ~1 bit；INL 正弦项杂散、ZOH 更新时钟镜像均与闭式预测吻合（1–2 dB 内，测试断言）。
- **AM/PM skew**（ex04）：160 MHz 下 0.5 ns skew 即把 EVM 从 −39.5 打到 −21.8 dB；互谱相位斜率估计精度 0.05 采样，校正后完全恢复——极坐标架构最尖锐的损伤，就是这条链的核心卖点。
- **带宽扩展**（ex02）：80 MHz OFDM 极坐标分解后包络/相位路径 99% 占用带宽 2.1×/3.2×；幅度域 hole punching 的 EVM 代价闭式可算（测试断言到 0.01 dB）。
- **π 跳变与直通 DAC 范围**（ex05，M2）：8DPSK 裸相位轨迹需要 fs/2=16 MHz 直通范围；2 MHz DAC 只削 1% 采样但环路"宿醉"使 DEVM 崩到 40%、ACP 恶化 ~20 dB（粗 mask 反而仍 PASS——**ACP 才是敏感指标**）；轨迹侧 2 MHz 斜率限制（矢量 hole punching）以 −32 dB 轨迹代价换回 DEVM 2.4%/零削波。OFDM polar 相反：LTE20 相位斜率 P99≈2×BW，直通 DAC 必须整体覆盖（test_lte_chain 固化）。
- **polar DPD 与两点校准**（ex06，M2）：AM-AM/AM-PM 逆 LUT 使 LTE EVM −37.2→−53.3 dB、ACLR1 −44.5→−62.6 dBc；由链路观测量拟合的 LUT 与精确模型反演只差 0.5 dB。离线两点增益估计一发命中（注入 3%，估计 3.00%，EVM 恢复到噪声极限）。RX 频段噪声解析预算 −143 dBc/Hz@45 MHz 量级。

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

- **M2 — 已完成**：LTE 20 MHz 全链路（fft/信道带宽解耦 numerology、E-UTRA ACLR1/2、风格化 SEM）、polar DPD（精确反演 + 测量拟合）、离线两点增益估计、直通 DAC 范围模型 + 矢量 hole punching、BT ACP、RX 频段噪声解析预算。
- **M3 — 已完成**：5G NR FR1/FR2 链路（38.104 numerology、NR OBUE/ACLR）、开环 DTC 增益/INL LUT 校准（CW 训练两次迭代，INL 杂散 −47→−92 dBc，残差达量化地板）、两点增益**在线** sign-sign LMS（event 引擎逐周期挂钩，5% 误差收敛到 0.1% 内、EVM 回到匹配噪声底）、DPA 交织（首镜像梳齿抑制 >15 dB 并推到 N×f_dpa）、post-DPA 记忆效应挂钩（线性记忆被 per-tone 均衡吸收的教科书行为有测试固化）。
- **T1–T3（评审补强轮）— 已完成**：CI workflow（多平台 pytest + offscreen GUI + iverilog job）与 Windows exe 打包 workflow；**DPA 效率模型**（SCPA class-D 律，polar 的核心卖点定量化）；三个**文献级对标**；**供电推压 AM→PM**（静态 LUT 与 GMP-ILA 均无法修复的损伤，测试固化）；response↔event **PSD 级回归**；BLE 分数信道；DTC dither 码路径 **RTL**（Verilog≡整数金向量≡浮点引擎恒等式，三方逐位一致）；**并行 Monte Carlo**（spec 化、进程池、扩展抽取）；配置 JSON 序列化（GUI 存取）；**EDR 整包**时序与分段指标；**功率 ramp**（max-hold 瞬态 ACP 度量——Welch 平均看不见 keying 瞬态，这本身是个教训）。
- **M4 — 已完成**：
  - **功率检波 skew 校准**（`estimate_skew_by_acp`）：只用带外功率观测的试探延迟搜索+抛物线细化，2.3 ns 注入恢复到 0.25 ns 内——芯片上只有功率检波器时的现实方案。
  - **整链 ILA-GMP 记忆 DPD**（`cal/memory_dpd.py`，vendored ILA）：把整条极坐标链（压缩 DPA+AM-PM+线性/三次记忆）当黑盒 PA 拟合，EVM −19.8→**−68.8 dB**、ACLR −27.2→−54.2 dBc（52 系数）。关键发现：链路满量程必须固定（`fs_scale_fixed`），逐次归一化会让"PA"非静态、ILA 收益封顶在 ~4 dB（测试固化）。
  - **相位插值形状研究**：固定斜率上限下 smoothstep 因窗口加宽 1.5× 反而比线性插值差（轨迹 EVM 与 ACP 双输）——诚实结论入测试。
  - **Monte Carlo 良率**（`polartx.montecarlo`）：逐芯片种子抽取 DPA 失配/DTC 增益/skew；参考案例 40 芯片 skew σ=0.5 ns：良率 5% → 逐芯片 skew 校准后 **70%**。
  - **双 GUI**：Streamlit 网页版（`gui/`，`streamlit run gui/Home.py`）与 PySide6 原生桌面版（`polartx.guiqt`，`polartx-gui` 入口，计算在工作线程、matplotlib 画布内嵌、offscreen 冒烟测试）——链路工作台/校准实验室/Monte Carlo 三页，计算全部在可脱离 GUI 测试的 `polartx.guiutil`。
  - **RTL 导出**（`polartx.export.rtl`）：polar DPD 双 LUT（12b 幅度/14b 有符号相位）定点化 → `$readmemh` ROM Verilog + 自校验 testbench + 金向量 CSV，iverilog 零失配验证（256 向量）。
- **暂缓**：GUI（Streamlit/Qt，姊妹库模式可直接照搬）、Monte Carlo 良率、RTL/AMS 导出。

## Examples

| 脚本 | 内容 |
|---|---|
| `ex01_ble_gfsk_adpll.py` | BLE 频率轨迹/眼图、EVM vs 两点失配设计图（双引擎）、Δf1/Δf2、BLE mask、相噪分解 |
| `ex02_polar_bandwidth_expansion.py` | 极坐标带宽扩展、包络零点、hole-punch 权衡 |
| `ex03_dtc_phase_modulator.py` | DTC 分辨率设计图（含 dither）、INL 杂散 vs 预测、ZOH 镜像、LO 相噪贡献 |
| `ex04_wifi_polar_chain.py` | WiFi 80/160/320 EVM/ACLR/Mask 总表、星座图、CFR CCDF、skew 灵敏度与校准闭环 |
| `ex05_edr_pi_jump.py` | π 跳变问题：直通 DAC 范围 × 轨迹斜率限制对 DEVM/ACP/mask 的联合影响 |
| `ex06_lte20_polar_chain.py` | LTE20 全链路：DPD on/off/拟合、两点校准、E-UTRA ACLR/SEM、RX 频段预算、DPA 特性反演 |
| `ex07_nr_polar_and_cal.py` | NR FR1/FR2 链路与星座、DTC LUT 校准前后频谱、在线两点 LMS 收敛轨迹、DPA 交织镜像 |
| `ex08_m4_dpd_mc_rtl.py` | ACP 搜索 skew 校准、整链 ILA-GMP 记忆 DPD、Monte Carlo 良率直方图、DPD LUT RTL 导出+iverilog |
| `ex09_measured_dpa.py` | **OpenDPD 真实 DPA 实测数据**：AM-AM/AM-PM 提取、静态极坐标 NMSE（≈−20 dB，与含记忆 GMP 的 −39 dB 差距即器件记忆）、实测特性入链 + polar DPD（−32→−50 dB）。需 `git clone --depth 1 https://github.com/lab-emi/OpenDPD.git ../OpenDPD` |
| `ex10_dpa_efficiency.py` | **效率——polar 存在的理由**：SCPA vs class-B 回退效率律、各预设调制平均效率（恒包络 85% → WiFi CFR8.5 43%）、CFR 深度的效率-EVM 权衡 |
| `ex11_benchmarks.py` | 文献级对标：Staszewski JSSC'05 EDGE polar（DEVM 1.9% vs 发表 ~2–3%）、Madoglio ISSCC'14 级 LTE-20（−31.2 vs ~−30 dB）、802.11n 级 20 MHz polar（−29.0 vs ~−28 dB） |
| `ex12_supply_pushing.py` | 供电推压 AM→PM：6 dB/倍频程律、静态 LUT 与 ILA 均无法修复（GMP 基不表达纹波积分相位）、BLE 分数信道扫描（EVM 平坦） |
| `ex13_packet_and_ramp.py` | EDR 整包（GFSK 头→guard→8DPSK 载荷分段指标）、功率 ramp 设计图（max-hold 瞬态 ACP：硬开关 −20 → 2 µs ramp −56 dBc） |
