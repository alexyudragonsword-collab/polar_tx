# 变更记录 / Changelog

按开发轮次记录。每轮写**做了什么**和**为什么**——README 的结果表给的是
现状，这里给的是到达现状的路径，包括几次结论被推翻的地方。

版本号尚未起用（`pyproject.toml` 停在 `0.1.0`），所以下面按里程碑而非
release 分组。

---

## 未发布 / Unreleased

### 文档补全（2026-08-15）

- 新增 `CONTRIBUTING.md`：把测试套件**已经在强制执行**的约定写下来——
  断言物理量而非快照、对标断言两侧、vendor 改编副本政策、"每个
  `bench_*` 必须能从 GUI 到达"、两个前端同步、EVM/mask 口径声明。每条
  都指名对应的测试或 CI job。
- 新增 `docs/architecture.md`：模块地图、三个数据流对象、两个会咬人的
  实现点（ADPLL 双引擎适用域、EVM 均衡口径）、"加东西改哪里"表。
- 新增 `LICENSE`（MIT，与 `pyproject.toml` 早已声明的一致）并说明
  vendored 代码沿用上游条款。
- 新增 `CLAUDE.md`：仓库级 agent 指引。
- docstring 缺失 87 → 24；剩下的是 GUI 内部件与父函数已解释的闭包。
- README 加文档索引；包结构树刷新（此前仍是 M1 版本）；测试数改成下界。

### 评审补强（2026-08-10）

四项此前挂起的审计项一次做完：

- **event 引擎适用域**：发现并固化——逐参考周期相位推进必须 ≪1 UI
  （BLE 0.8%、EDR 4.2% 可用，LTE 29% 不可用）。越界发 `RuntimeWarning`
  并在诊断给 `ui_per_ref_cycle_p99`。
  过程中提出并**自行证伪**了一个假说：LTE event 模式的偏差不是两点通路
  差一拍造成的（两个方向移位都更糟）。
- **SEM 框架重建**（`metrics/sem.py`）：限值按 mask 声明的 **RBW 积分**
  而非逐 FFT bin 比较，裁决不再随 `nfft` 漂移；**分开报告带内/带外裕量**
  ——带内相切把总裕量恒钉在 0.00 dB，`oob_margin_db` 才是随设计变化的
  数。`MaskSpec.source` 使"工程模板 vs 认证条款"机器可读。
- **vendored 行为测试**（`test_vendor_behaviour.py`，26 项）。写的过程中
  我自己的三个假设被证明是错的（不是 vendor 的 bug）：
  `quantize_symmetric` 用 2 的幂步长、顶码是钳位的、CCDF 范围受实际
  PAPR 约束——按真实契约改的是测试。
- **指标不再静默返回 NaN**：`metrics/dpsk.py` 在 burst 短于两端裁剪时抛
  异常。静默 NaN 会一路传到报告和 GUI 里显示成 "nan%"，没有任何线索。

### 全工程审计（2026-08-10）

六处修复：弱断言加强、硬编码计数改成自维护不变量
（`len(PRESETS) == 14` 和 GUI `== 3` 页曾各挡住过一次合法改动）、
Streamlit 指标表 pyarrow 混类型 traceback、examples 纳入 CI 全量运行。

### 管理层汇报材料（2026-08-04）

`slides/`：8 页中文 PPTX（`build_deck.js` 可重建）。沙箱里
LibreOffice 不可用，改用几何审计脚本 `qa_geometry.py` 检查溢出——并先用
一个故意溢出的盒子验证了检测器本身。

### EVM 口径与 AM/PM skew（2026-08-03）

一串互相咬合的问题，按提问顺序：

- **CPE 被处理掉了吗**——两种口径都不去除，但实测这些预设 CPE 仅
  0.01–0.6°（LO 锁相后相噪被高通整形到符号率以上）。报告直接给出
  `CPE rms [deg]`。
- **skew 恶化得那么快，和 CPE 有关吗**——无关。skew 是
  `y = x·[a(t−τ)/a(t)]`，数据相关、宽带，产生的是 ICI 不是 CPE。
  报告在 scalar/per-tone 差值 >1 dB 时自动补一行 per-tone EVM。
- **频谱仪会不会补偿掉**——不会。功率域测量没有均衡器，skew 的频谱再生
  **完全暴露**且比 EVM 更早报警：WiFi 160 MHz 0.2 ns 时 per-tone EVM
  还有 −34.4 dB，mask 已经 FAIL、ACLR 从 −58 掉到 −37.3 dB。
- **窄带里是不是完全没考虑**——是漏了。skew 是**包络路径**效应，随包络
  变化而非架构而定，所以给 `ble_adpll` / `lte20_adpll` 也补上了
  `env_skew_s`：LTE 1 ns 即失配（比 160 MHz 的 0.2 ns 宽容约 8 倍，与带宽
  成比例），恒包络 BLE 免疫。

### 星座图审计（2026-07-25）

- RFIC'26 对标星座散掉：两个原因叠加——均衡口径不匹配（双抽头链要
  per-tone）+ CFR 过猛（8 dB 把 4096-QAM 削成 EVM 地板，改 9 dB 落到
  −40.6 dB vs 论文 −40.7 dB）。
- 逐个预设审计后发现一个真 bug：**SC-FDMA 画图错误**——DFT 预编码后的
  频域符号根本不存在 QAM 星座，必须先反 DFT。另加 ≥1024-QAM 中心放大与
  星座判据。

### GUI 追平（2026-07-25）

库先行、GUI 落后：三页新功能（FIR+Doherty / 架构选择器 / RTL 导出）在两个
前端都补齐，RFIC'26 对标进注册表。加了"每个 `bench_*` 都能从 GUI 到达"的
不变量测试防止再犯。web GUI 用 `AppTest` 做了真实验证（此前只是"看起来
接上了"），顺手修掉 Arrow 序列化的坑。

---

## 里程碑 / Milestones

### 图标与打包（2026-07-24 ~ 07-25）

项目图标 → docs favicon + README 角标 + 四个 Windows onefile exe 的图标。

### 五项评审建议（2026-07-24）

- **CI 首次真跑**（此前只是文件存在）
- **vendor 漂移检查**（`tools/vendor_check.py` + `vendor-drift` job）：
  逐文件分类 verbatim/adapted/extended/extracted，只对被批准的本地改动钉
  哈希。首次上 CI 39 个文件报 "pinned commit not found"——把姊妹仓 checkout
  钉到确切 SHA，并让检查器在 pin 不可达时 **skip** 而不是误报 DRIFT。
- **架构选择器**（`selector.py`）：第一版模型给 BLE/LTE 推荐 DTC（反了），
  在"共用综合器"基础上重建 + 补 DTC 特有噪底才对。
- **更全 RTL/AMS 导出**：温度计译码器（127 段的金向量温度计字需
  2^127−1，溢出 int64，改用 Python 大整数 + 十六进制金向量）、CFR 削波级、
  DTC 相位累加器、DPA Verilog-AMS `wreal` RNM 模型。
- **多核 / Doherty 合路**（`dpa/combiner.py`）：效率由负载调制物理导出而非
  拟合；WiFi 80 MHz 1024-QAM 平均效率 43%→58% @ 同 EVM。

### F1–F4 波形保真（2026-07-24）

SC-FDMA/DFT-s-OFDM（LTE 上行默认，PAPR 低 1.7 dB → 平均效率 35.9→41.4%）、
BLE 认证口径（Δf2max 符号内最大值、调制指数容差、漂移）、EDGE 数值提取的
线性化 GMSK C0 脉冲、导频 + CPE 跟踪 + 前导信道估计的接收机式 EVM。
**信道编码刻意排除**。

### 文献对标（2026-07-24）

Staszewski JSSC'05 EDGE → Madoglio ISSCC'14 LTE-20 → Ben Bassat JSSC'20
WiFi 6 → Degani RFIC'24 WiFi 7 → Borokhovich RFIC'26 FIR+Doherty MLO。
每个都断言两侧，落在发表的量级区间内。

### T1–T3 评审补强轮（2026-07-24）

CI workflow + Windows exe 打包；DPA 效率模型（polar 的核心卖点定量化）；
供电推压 AM→PM（静态 LUT 与 GMP-ILA **均无法**修复，测试固化）；
response↔event PSD 级回归；DTC dither RTL 三方逐位一致；并行 Monte Carlo；
配置 JSON 序列化；EDR 整包；功率 ramp（max-hold 瞬态 ACP——Welch 平均看不见
keying 瞬态，这本身是个教训）。

### M4–M6（2026-07-24）

- **M4**：ACP 搜索 skew 校准（只用带外功率观测）、整链 ILA-GMP 记忆 DPD
  （EVM −19.8→−68.8 dB；**关键发现**：满量程必须固定，逐次归一化会让"PA"
  非静态、收益封顶在 ~4 dB）、Monte Carlo 良率、Streamlit GUI、RTL 导出。
  另一条诚实结论入测试：固定斜率上限下 smoothstep 因窗口加宽 1.5× 反而
  **输给**线性插值。
- **M5**：OpenDPD 实测数据通路 → 测量定标 DPA；`docs/index.html` 图文设计
  指南（13 节，中英双语）。
- **M6**：PySide6 原生桌面 GUI（`polartx-gui`）。

### M2–M3（2026-07-24）

- **M2**：LTE 20 MHz 全链路、polar DPD（EVM −37.2→−53.3 dB）、离线两点增益
  估计、直通 DAC 范围模型 + 矢量 hole punching、RX 频段噪声预算。
  π 跳变实验的结论：粗 mask 仍 PASS 而 ACP 恶化 ~20 dB——**能骗人的是
  EVM，说实话的是 ACP**。
- **M3**：5G NR FR1/FR2、DTC 增益/INL LUT 校准（INL 杂散 −47→−92 dBc）、
  两点增益在线 sign-sign LMS、DPA 交织镜像、post-DPA 记忆效应。

### Step 0–3：首期（2026-07-24）

脚手架 + vendor 移植（`pll_simulator@d7be4712`、`PA_DPD@44f9ee99`）→
窄带 BLE GFSK + ADPLL 两点链路 → 宽带 WiFi OFDM + 开环 DTC 链路 → README。

`bt_edr_adpll` 的 `mode` 参数在此期间改名为 `dpsk`：它与 response/event
引擎选择的 `mode=` 撞了。
