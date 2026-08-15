# polar_tx 架构说明 / Architecture

写给要**改这个库**的人。只想用的人看 `README.md` 和
[`docs/index.html`](index.html)（图文设计指南）；写代码的约定看
[`CONTRIBUTING.md`](../CONTRIBUTING.md)。

---

## 1. 一句话结构

整个库只有**一条**信号链。两种架构（窄带 ADPLL 两点、宽带开环 DTC）不是
两条链路，而是同一条链路里换掉一个对象：

```
Waveform ─→ [CFR] ─→ polar_split ─┬─ 包络 u(t) ─→ 归一化/DPD/skew/量化 ─┐
                                  │                                    ├─→ DPA(code, θ) ─→ y ─→ 指标
                                  └─ 相位 θ(t) ─→ PhaseModulator ──────┘
```

`PhaseModulator` 是唯一的架构开关：

| 换成 | 得到 | 典型应用 |
| --- | --- | --- |
| `ADPLLTwoPoint` | 窄带 polar TX | BLE、BT EDR、LTE ≤20 MHz |
| `DTCPhaseModulator` | 宽带 polar TX | WiFi 6/7 ≤320 MHz、5G NR ≤200 MHz |
| `IdealPhaseModulator` | 调试基准（相位直通） | 分离幅度路径的问题 |

**这个设计是有代价换来的收益**：因为其余每一级（CFR、分解、包络量化、
skew、DPA、指标）都是共用的同一份代码，窄带和宽带的结果可以直接比较，
不需要"两边口径不同"的免责声明。改动共用级时请保持这个性质。

## 2. 模块地图

```
src/polartx/
├── waveforms/        激励：base.Waveform 是所有下游代码看到的唯一形状
│   ├── ofdm.py       通用 OFDM 引擎（SCS 可配；WiFi 预设与 padpd 逐位一致）
│   ├── ble.py        GFSK（BT=0.5, h=0.5），恒包络
│   └── edr.py        π/4-DQPSK / 8DPSK（EDR），含线性化 GMSK C0 脉冲
├── polar/split.py    极坐标分解/重构 + 带宽扩展分析 + hole punching
├── phasemod/         ★ 架构开关
│   ├── base.py       PhaseModulator ABC + PhaseModResult(phase_out, diagnostics)
│   ├── adpll_tp.py   两点调制，双引擎（见 §4）
│   └── dtc_openloop.py  开环 DTC：量化/dither/增益误差/INL/抖动/LO 相噪
├── dpa/              数字 PA
│   ├── dpa.py        码表化 DPA（预计算后向量化）
│   ├── mismatch.py   温度计+二进制阵列失配 → 幅度表/INL/DNL
│   ├── characteristics.py  AM-AM/AM-PM：ideal | scpa | lut
│   └── combiner.py   Doherty / 多核合路
├── chain.py          ★ ChainConfig + PolarTX + PolarResult（见 §3）
├── fir.py            双抽头 FIR TX（RFIC'26 类 MLO 陷波），与主链同口径
├── impairments.py    分数延迟 skew 注入、ZOH、包络量化
├── cal/              校准：skew、两点增益（含在线 LMS）、DTC LUT、polar DPD、GMP 记忆 DPD
├── metrics/          指标：EVM/ACLR/PSD/mask/SEM/BLE Δf/DPSK dEVM/接收机式 EVM/RX 频段噪声
├── analysis/         解析对照：环路响应、量化噪底、ZOH sinc 与镜像、噪声预算
├── measured.py       实测数据通路（OpenDPD 格式）→ 测量定标的 DPA 模型
├── montecarlo.py     良率分析（spec 化、进程池并行）
├── export/rtl.py     定点化 + Verilog/Verilog-AMS 导出 + 金向量
├── selector.py       架构选择器：给定需求，DTC vs ADPLL 哪个更合适
├── presets.py        ★ 端到端预设（标准链路 + 文献对标）
├── guiutil.py        ★ 两个 GUI 共用的全部计算（可脱离 GUI 测试）
├── guiqt/            PySide6 桌面 GUI
└── vendor/           改编移植区（见 §5）
```

带 ★ 的是新人最先要读的五个文件。

## 3. 数据流三个对象

**`Waveform`**（`waveforms/base.py`）——激励和它的"真值"打包在一起：
`x`（复基带）、`fs`、`bw`、`kind`（`"ofdm"` / `"gfsk"` / `"dpsk"`）、
`ofdm_ref`（参考星座网格）、`meta`（numerology）。下游按 `kind` 分派，
所以加一种新调制 = 加一个 `kind` + 在 `metrics` 里给它一个评分函数。

**`ChainConfig`**（`chain.py`）——不属于相位调制器、也不属于 PA 的一切。
关键在于它是**架构无关**的：同一个 `ChainConfig` 驱动两种架构。最常用的
几个旋钮（`env_skew_s`、`env_floor`、`cfr_papr_db`、`fs_scale_fixed`、
`phase_slew_max_hz`）在 dataclass 的 docstring 里有量化说明。

**`PolarResult`**（`chain.py`）——一次运行的输出 **加上每一级的中间抽头**，
什么都不丢：`env_cmd` vs `env_code` 看幅度量化，`phase_cmd` vs `phase_out`
单独看相位调制器，`info["phasemod"]` 拿调制器自己的诊断。指标方法
（`evm/aclr/psd/check_mask/avg_efficiency`）按 `wf.kind` 分派。

## 4. 两个需要知道的实现要点

### 4.1 ADPLL 两点调制的双引擎

`ADPLLTwoPoint` 有两个模式，互为交叉验证：

- **`mode="response"`**（快，主力）：在基带 FFT 网格上构造两点复合响应
  `H_tp(f) = H_lp + dp_gain·(1+kdco_err)·H_hp`，整帧 FFT 滤波；相噪由
  `analyze()` 的总 PSD 合成叠加。整帧秒级。是**线性化**模型，不含
  ΣΔ dither × 调制这类非线性耦合。
- **`mode="event"`**（慢，真值）：把频率轨迹重采样到 `fref` 网格，逐参考
  周期跑 vendored ADPLL 引擎。

**event 引擎有适用域**：每参考周期的相位推进必须 ≪ 1 UI。实测
BLE 0.8%、EDR 4.2% 没问题，LTE 29% 就不行。越界时会发 `RuntimeWarning`
并在诊断里给 `ui_per_ref_cycle_p99`。凡是杂散类结论，一律以 event 模式或
解析预测背书（并固化成测试）。

### 4.2 EVM 均衡口径

链路默认 **scalar** EVM（保留频率相关/记忆型失真可见），真实 VSA 按标准做
**per-tone** 均衡（802.11 用 LTF 信道估计，3GPP 用 RS FDE）。两者只在存在
频率响应的损伤上分叉——AM/PM skew 0.5 ns 时差 5.6 dB。

因此 `PolarResult.evm_equalize_default` 声明了这个结果用哪种口径，报告层
读它，**保证画出来的星座图和打印的 EVM 是同一个口径**。加新的结果类型
（例如 `fir.py` 的双抽头链用 `per_tone`）时必须设这个属性。

CPE **两种口径都不去除**；这些预设里 CPE 只有 0.01–0.6°（LO 锁相后相噪被
高通整形到符号率以上），报告直接给出 `CPE rms [deg]`。

## 5. Vendored 代码

`vendor/pllsim/`（相位路径）与 `vendor/padpd/`（幅度路径与指标）是两个姊妹
仓库的**改编副本**，不是 git 子模块也不是 pip 依赖。每个文件头注明出处
commit 与路径，`tools/vendor_check.py` + CI 的 `vendor-drift` job 每次 push
校验漂移。改 vendored 文件的规矩见 `CONTRIBUTING.md` §3。

## 6. 加东西时应该改哪里

| 想加 | 改哪里 | 别忘了 |
| --- | --- | --- |
| 新调制/新制式 | `waveforms/` 加生成器；`metrics/` 加评分 | `presets.py` 加预设，`metrics/masks.py` 加模板 |
| 新相位调制架构 | `phasemod/` 实现 `PhaseModulator` | docstring 写清适用域并在越界时告警（§4.1） |
| 新损伤 | `impairments.py` 或 `ChainConfig` 加字段 | 单调性/标度律测试，字段注释写单位 |
| 新指标 | `metrics/` | 声明测量口径；输入不足要抛异常，不返回 NaN |
| 新对标预设 | `presets.py` 的 `bench_*` + `polartx.__all__` | 必须进 `guiutil.PRESETS`（有测试挡） |
| 新 GUI 页面 | `guiutil.py` 放计算 | **两个前端都要接**，两边都要有测试 |

## 7. 已知边界

- `mode="response"` 是线性化模型；杂散结论不以它为准。
- 各制式 mask/SEM 是**工程化模板**（`source="stylized"`），不是认证限值；
  `MaskSpec.is_conformance` 为此存在。
- 远端噪底（如 −160 dBc/Hz @ 80 MHz）超出可行记录长度的 Welch 分辨能力，
  这类指标以解析预算（`analysis/responses.py`）为主。
- 信道编码刻意排除：TX 损伤链路上没有任何指标在比特映射之后测量。
