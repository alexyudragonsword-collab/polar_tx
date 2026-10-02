# 变更记录 / Changelog

按开发轮次记录。每轮写**做了什么**和**为什么**——README 的结果表给的是
现状，这里给的是到达现状的路径，包括几次结论被推翻的地方。

版本号尚未起用（`pyproject.toml` 停在 `0.1.0`），所以下面按里程碑而非
release 分组。

---

## 未发布 / Unreleased

### 慢状态记忆从"完整源"反演（记忆反演阶段 2）（2026-10-02）

`fit_residual_memory(ch, source=...)` 接 PA_DPD"完整源"npz（`load_complete_npz`）：
`step` 组离线辨识增益调制 τ，`burst` 组过静态模型后训练残差 `StateConditionedSpline`，
状态 α 取辨识值；缺组时报错并点名缺哪组，step 无显著调制时拒绝拟合。一键形式
`dpa_from_complete_source(path)`；无硬件时用 `synthetic_thermal_source`——vendored
自加热虚拟 DUT（新 vendor `pa/thermal.py`、`drift.py`、`reference_pa.py`，逐字节）
录出主 / burst / step 三组并写盘读回。报告新增 `taus_heat_s`、`state_alphas`、
`fast_only_nmse_db`（同一对数据上无状态的 SplineGMP）、`state_gain_db`，以及
`rank_deficiency` / `dead_columns`。

验收全部是测试（`tests/test_slow_memory.py`，不需要数据、不 skip）：

| 项 | 实测 | 门槛 |
|---|---|---|
| 离线 τ（真值 5 / 30 µs） | 5.13 / 29.59 µs | 误差 ≤ 10% |
| burst：状态残差 vs SplineGMP 残差 | −31.91 vs −23.33 dB（+8.6） | ≥ 6 dB |
| 预热平稳主采集：两者之差 | 0.12 dB | ≤ 1 dB |
| 缺 burst / step | 报错并点名 | — |
| 入链（DPA + 状态记忆，冷启动）对实测 burst | −32.0 dB（模型自身 −31.9） | 差 < 1 dB |

三个过程中被实测改掉的认识，详见 `cairn/measured-memory.md`：**平稳对照要求主采集预热
且够长**（冷启动 54 µs 的主采集是加热暂态，状态模型假性好 5.4 dB；PA_DPD 示例文件就是
这样录的，已在 PA_DPD 数据接口文档里写明）；**虚拟 DUT 自带 −34.9 dB 地板**（分块推进时
FIR 每 128 采样重启，块首三个采样出错；平稳残差 −34.0 正落在上面，所以阶段 2 的 NMSE
受 DUT 限制、8.6 dB 是下界——上游问题，vendored 副本未改）；**状态样条按结构秩亏**（单位
分解 + 共用 `q_scale`，180 系数秩亏 49），条件数改报张成空间上的值（1.6e4），满秩模型
上与旧值完全相同。τ 用错的代价也量了：×0.3～×3 增益不变，×10 太慢时反而差 12 dB。

另：更正阶段 0 的记录——padpd pin 44cbcb3 在 vendor 时是 PA_DPD
`claude/digital-polar-tx-dev-r0c338` 的头，该分支后来被删过；现在可从
`claude/lucid-einstein-58n326` 到达，仍不在 `main`。`vendor/__init__.py`、architecture §5、
CI 注释改为"分支名会变、SHA 不会"的写法。

### 快记忆从实测反演：残差模型进 `chain.memory`（记忆反演阶段 1）（2026-10-01）

`measured.py` 三个新函数：`static_prediction(ch, x)`（静态 LUT 模型的输出，
与 `static_nmse_db` 同一份代码）、`fit_residual_memory(ch, model_factory)`
（在"静态输出 → 实测输出"的残差对上拟 padpd `PAModel`，默认 GMP-510；时间顺序
前 80% 训后 20% 验，不打散；报告分开给静态 / 无记忆残差 / 完整残差三个验证集
NMSE、系数数、条件数）、`load_measured_dpa(name, with_memory=True)`（`ch["memory"]`
直接作 `PolarTX(memory=...)`，`ch["fs_scale"]` 是调用方必须填进
`fs_scale_fixed` 的实测满量程）。`chain.py` 只改了 `memory` 参数的 docstring。

两个口径决定，都在 `cairn/measured-memory.md`：**残差模型在链路 y 的尺度上拟**
（训练对除以 `chain_gain`，链路名义增益为 1，非线性项不是尺度不变的）；
**对齐只做整数延迟**（小数群延迟是线性记忆，重采样掉它会塞进一个 sinc 让模型去
抵消：合成 DUT 复合 −46.6 vs −52 dB）。后者让 160 MHz 静态 NMSE 从 −19.93 变成
−19.95，现有断言不动。

实测（OpenDPD，验证集，chain 尺度）：

| 器件 | 静态 LUT | +无记忆 5 阶 | +GMP-510 残差 | +SplineGMP 残差 | PA_DPD 直拟 / 发表 |
|---|---|---|---|---|---|
| DPA_160MHz | −19.95 | −19.97 | **−37.5** | −37.7 | −38.7 / −39.2 |
| DPA_200MHz | −20.54 | −20.56 | **−33.3** | — | −33.8 / −33.7 |

无记忆残差只买 0.02 dB：**"19 dB 就是器件记忆"这句话成立**，静态模型没漏掉任何
静态形状。入链（WiFi-160、峰值放到实测满量程、同采样率）：极坐标 DPD 在静态器件上
−32.0 → −49.9，记忆进链后 −19.75 → −19.94（0.2 dB）；整链 ILA（GMP-510 结构）
−19.75 → −36.5（16.7 dB；默认 ILA 结构只有 13 dB；ACLR 反而 −38 → −34，ILA
在这里拿频谱换 EVM，如实记）。模型 npz 往返后入链输出逐位相同。

与方案的两处出入：SplineGMP 条件数只比 GMP 低 1.4 个量级（2.1e5 vs 8.8e3），
不是 2–3 个——残差输入已是压缩包络，多项式基没那么共线，测试按实测卡 ≥ 1 个量级；
合成 DUT 的复合 −52 dB 要靠整数对齐，静态提取本身在纯静态器件上的地板是 −35 dB
（顶部稀疏箱钳位，与箱数无关），阶段 1 没动它。ROADMAP B2 改写为"快记忆已反演，
慢状态待完整源容器"。

### padpd 子树重新 vendor 到 44cbcb3（记忆反演阶段 0）（2026-10-01）

`vendor/padpd/` 整体从 `44f9ee99` 推进到上游主干 `44cbcb3`。推进前先把三个
`pa/*.py` 与钉定 blob 逐字节对照：**零差异**，所谓 42/15/20 行差异全是上游前进，
本地没改过，所以直接重拷，不走 manifest 也不需要回游。上游带来的能力：
`lstsq_fit` 的加权与结构惩罚（keyword-only，默认不变）、`basis_cond`、MP 的
`gain_curve`、ILA 持久化恢复工厂、PSD 有限地板。新 vendor 四个文件——
`pa/spline.py`、`pa/spline_state.py`（B 样条 MP/GMP 与慢状态调度）、
`gain_modulation.py`（阶跃探针 τ 辨识）、`data/complete.py`（完整源容器）——
以及抽取版 `pa/presets.py`（只取 `gmp_opendpd_510` / `mp_opendpd_500`）；
`pa/__init__.py` 的 `load_model` 注册表扩到全部 vendored 模型类。

验证按 dump-比对协议：ILA 链、OpenDPD 静态提取、PSD/CFR/对齐输出的数字与哈希
在重拷前后**逐字节相同**。`tools/vendor_check.py` 新增 `pin_frozen`：pllsim 两条
故意停留的 pin 报 *frozen* 而非 *stale*，`--strict` 从"永远红"变成可用的门。
加跨库往返测试：PA_DPD 侧写的 GMP npz 由 vendored `load_model` 读回逐位同输出。

### Outphasing 两路分支共用一个 LO（ROADMAP B7 收口）（2026-09-30）

阶段 3 记下的缺口：`OutphasingTX` 每路分支各抽一份 LO 相噪，真机两路 DTC 挂在
同一个 PLL 上。修法没碰 `chain.py`：`DTCPMConfig` 加 `lo_pn_seed`（None = LO 相噪
跟随运行 seed、逐位不变；整数 = LO 样本走自己的流 `default_rng([seed, 1])`，
与抖动流分离），`OutphasingTX.run` 每次运行浅拷贝分支链，把 `lo_pn_seed` 钉成
本次 seed——两路分支 LO 样本相同、DTC 抖动与 dither 仍各自独立；无 LO 模型的
调制器（Ideal / ADPLL）不受影响，`shared_lo` 字段与 `info["shared_lo"]` 报告是否生效。

**更正阶段 1 和阶段 3 的判断**：阶段 1 把 outphasing 比极坐标差 3 dB 归因于
"两分支满功率、相噪不随包络回退且两路独立不相消"；阶段 3 估计其中约 3 dB 来自
独立 LO。实测是**整个差距都来自独立 LO**：共用 LO 后 8 符号 burst 上 outphasing
−39.84 dB vs 极坐标 −40.00 dB（地板口径差 0.4 dB），4 符号上 −0.3 dB，与选型器
共模 LO 预算 0.44 dB 一致；独立 LO 时地板差 5.1 / 4.0 dB。物理上说得通：这条链
的 EVM 地板由 LO 相噪主导（−49 dB 项），DTC 量化/抖动地板（−67 / −55）即使各
多付 5.5 dB 也不露头。ACLR 随之 −49 → −53 dBc，分支相位 1 dB 容忍度 0.74° → 0.51°
（自身地板更低，同一失配更显眼）。ex19 / ex20 / README 表同步。

### 选型器四候选 + 三架构同台表（阶段 3：同台对比）（2026-09-30）

`selector.py` 从两候选（ADPLL / DTC）扩到四候选：`_score_outphasing`——DTC 地板
逐项加 10log10(PAPR/2) dB（两路分支恒满幅，独立噪声按功率比放大；共模 LO 不加，
`outphasing_shared_lo=False` 时也加），再加分支相位失配闭式项
`20log10 δ + 10log10((PAPR−1)/4) + quad_inband_db`（带内占比 −7.5 dB 是在
WiFi 160 MHz 波形上按 `phase_mismatch_evm_budget_db` 实测的常数，测试对照 1°/2°
差 < 0.1 dB）；`_score_rfdac`——两轴量化 2Δ²/12 摊到过采样率上（6 bit 链路实测
−41.5 vs 闭式 −41.1）、LO 抖动、镜像 −IRR。每个候选新增 `eta_avg`：各自效率律
（SCPA / Chireix / iq_cells 相位平均 4a/π）在截断瑞利包络分布上积分，与链路
实测差 < 0.01（0.428/0.402/0.270 vs 0.428/0.400/0.280）。

**排序规则（本阶段决定）**：不可行垫底 → 达标者优先 → 达标者中效率最高 →
EVM 最低；`Requirement.eta_avg_min` 把效率变硬门槛。理由：达标之后的 EVM
裕度不值钱，效率值钱。验收：BLE 仍选 ADPLL；320 MHz 4096-QAM 不选 outphasing
（比 DTC 差 7.3 dB 且效率更低）；`run_selector_report` 三个前端同步拿到四行
和效率、交叉图四条曲线，`appbridge.py` 未动。

阶段 3 的对照测试揪出两个建模缺口：**RF-DAC 链没有 LO 相噪**——阶段 2 的
`RFDAC` 只有 50 fs 抖动，同台表里 −42.4 dB 比选型器乐观 15 dB。已补
`RFDACConfig.lo_pn / lo_loop_bw`（与 `DTCPMConfig` 同字段同生成器），
`wifi_rfdac` 带上同一计划的 LO；同台表 RF-DAC 改为 **−40.0 dB**，与极坐标
相同——两者都被同一个 LO 限住。**outphasing 链两路 LO 相噪独立抽**（真机共模），
链路里 ~3 dB 是它造成的；修它要碰主链，记入 ROADMAP B7，选型器默认按共模打分。

`ex20` 表新增"最敏感旋钮的 1 dB 容忍度"列（4 符号 burst 二分）：极坐标 skew
0.03 ns、outphasing 分支相位 0.74°、RF-DAC I/Q 相位 0.62°。`ex16` 决策表与
交叉图同步四候选。

### 笛卡尔（RF-DAC）发射机拓扑（阶段 2：第三种架构）（2026-09-30）

第三种发射机拓扑：数字 I/Q 发射机。`rfdac.py` 的 `RFDAC` 是 I/Q 两组电流单元
阵列（符号-幅度码，I 阵列用 `seed`、Q 阵列用 `seed+1`，单元失配**复用**
`dpa/mismatch.py` 的 `code_amplitude_table`），带笛卡尔架构特有的损伤：I/Q
增益/相位失衡（镜像，`iq_image_rejection_db` 给精确 IRR）、LO 泄漏（DC 项）、
时钟抖动（DTC 口径 2π·fout·σ_τ，受 `noise` 门控）；效率律 `("iq_cells", η_peak)`：
负载拿 I²+Q²，阵列付 |I|+|Q|。`cartesian.py` 的 `CartesianTX(cfg, rfdac)` 吃同一个
`ChainConfig`，但只读 `cfr_papr_db`；其余十个包络/相位路径字段被设成非默认值时
**告警**（`UserWarning` 点名字段）而不是静默无效——这是本仓反复踩的"旋钮看着接了
其实没接"的反面。`CartesianResult` 自带同名指标方法（没有极坐标运行可借），
`wifi_rfdac` 预设带 `.tx` 别名进注册表 "WiFi 160 MHz (RF-DAC)"。
`chain.py`、`ChainConfig`、`PolarResult` 仍一行未改。

五条验收全部是测试（`tests/test_rfdac.py`、`tests/test_cartesian_chain.py`）：
无失配 14 bit 下 EVM 与极坐标理想链**相差 0.001 dB**（6 bit 差 3 dB、16 bit 无变化，
证明地板是 CFR 的）；0.1 dB/1° 失衡的单音镜像与解析 IRR **精确一致**（39.61 dB，
与教科书 4/(g²+φ²) 差 0.01 dB），调制链上 EVM 退化贴着 −IRR；同 `(n_bits, n_thermo,
sigma_cell, gradient, seed)` 下 I 阵列的 INL/DNL 向量与 `DPA.inl_dnl()` **逐元素相等**，
失配误差功率随 σ_cell **20 dB/十倍**（6+4 分段 10 bit 在 1% 时低于 CFR 地板 30 dB，
所以不动 EVM——断言的是误差本身）；6 dB 回退 RF-DAC 轴上 42.5% < SCPA 50.9%，
同一 burst 平均 28.3% < 42.8%；`env_skew_s ≠ 0` 触发告警且输出**逐位相同**。

`ex20_three_topologies.py` 三架构同台（真实预设，加噪）：极坐标 −40.0 dB / −58 dBc /
42.8%；outphasing −37.1 / −49 / 40.0%；RF-DAC **−42.4 / −60 / 28.0%**。RF-DAC 线性
最好、效率最差，正是这三种拓扑的教科书取舍。已知取舍：效率律不含单元充放电项
（乐观的 RF-DAC 律）；`backoff_db` 以轴上满码为基准，对角峰可高出 √2。

### Outphasing 发射机拓扑（阶段 1：验证抽象）（2026-09-30）

第二种发射机拓扑，与极坐标链**同波形、同 CFR、同相位调制器、同 DPA 模型、
同指标**对比。不加第二个架构开关：照 `fir.py` 的先例，`outphasing.py` 的
`OutphasingTX` 是包住 `PolarTX` 的兄弟类，把信号分解成两路恒包络分支
（φ±θ，θ = arccos(A/A_max)），各跑一次同一条链（DPA 满码），在
`dpa/combiner.py` 新增的 `OutphasingCombiner`（isolated / Chireix）里合路；
`OutphasingResult` 借 `_as_polar()` 复用 `PolarResult` 全部指标；
`wifi_outphasing` 预设带 `.tx` 别名进注册表 "WiFi 160 MHz (outphasing)"。
`chain.py`、`ChainConfig`、`PolarResult` 一行未改。

五条验收全部是测试（`tests/test_outphasing.py`，实测值写在断言旁）：
理想链下 EVM 与极坐标理想链**相差 0.000 dB**（分解恒等式 s1+s2=x 逐位成立）；
分支相位失配 1° 的 EVM 退化与闭式预算差 **0.35 dB**、随角度单调；
isolated 平均效率 = η_pa·E[cos²θ]（相对误差 1e-9 内）、Chireix 高于 isolated
（43.9% vs 10.9%）；报告 θ 分布，θ>80° 占比 **21%**（去掉 CFR 更高）；
`env_skew_s` 对 outphasing 输出**逐位无影响**，而同一旋钮让极坐标退化 20 dB。

`ex19_outphasing_vs_polar.py` 的同台表（真实 DTC 11 bit + Rapp DPA，加噪）：
极坐标 EVM −40.0 / ACLR −58 / 效率 42.8%；outphasing −37.1 / −49 / Chireix 40.0%、
isolated 10.0%。EVM 差 3 dB 的物理原因：两路分支始终满功率，相位噪声不随包络
回退，且两路独立不相消。已知取舍：Chireix 的负载调制只进效率律，信号路径与
isolated 相同（有限 PA 输出阻抗带来的幅相失真留待后定）。

### 静态检查闸门 + 文档锚定 + 拆报告层（2026-09-12）

一次工程体检之后的三批整改。体检本身的结论是**主干是健康的**——277 项测试
0 失败、自身代码语句覆盖率 92.8%（vendor 57.3%）、vendor 0 漂移、bandit 无
中高危——短板全在工程外围，所以这一轮动的也全是外围。

**加了两个闸门，它们立刻抓到东西。** 仓库此前没有任何静态检查配置。ruff
（CI `lint` job，版本钉死）抓出 3 处真的引用丢失：一处 `paths = emit_dpd_rtl(...)`
的返回值从来没被读过、一处重写后遗留的 `prev = -np.inf`、两个导入了却没用的
preset。mypy 抓出的 9 个非 vendor 错里 6 个是同一个根因——`Waveform.ofdm_ref`
和 `FIRTxPreset.fir_tx` 用 `object` 当"其实是某个具体类"的占位，于是**穿过这两个
字段的一切**也不可检查。改成 `if TYPE_CHECKING:` 导入真实类型后，顺出两个潜在
真问题：vendored config 允许 `seed=None`，而 `cfg.seed + 777` 会让带 pilot 或
preamble 的波形死在算术里；蒙卡穿过抽象 `PhaseModulator` 取 `.cfg`，而
`ADPLLTwoPoint` 根本没有这个属性（在 `.pll.cfg`）。两处都改成明确窄化 + 抛带话
的异常。范围取舍（vendor 在两个工具里待遇相反、哪些风格码不管）见
`CONTRIBUTING.md` §8.1。

**`docs/architecture.md` 补到文件级并被测试卡死。** `cal/`、`metrics/`、`guiqt/`
之前只有目录级的一行概括，14 个模块的文件名在这份"模块地图"里一个都查不到。
新增 `tests/test_docs_consistency.py` 卡两个方向（模块没进地图 / 地图写了不存在
的文件），顺带把 README 的 "230+ 项测试" 换成可核对的测试函数数——它按 AST 数
函数而不数 pytest 实收，因为实收数取决于装了哪些可选依赖，那是机器的属性。
四个检查都做了变异验证。

**拆了 `run_chain_report`**：168 行、圈复杂度 D(29) → A(3)，计算与绘图分离，
星座图的点由指标阶段算出，图阶段只负责画——图上画的和旁边印的 EVM 不可能来自
不同的均衡口径。验证不靠"套件还绿"（套件里没有任何测试会因为图变了而失败）：
16 个 preset ＋ 3 个带损伤配置的指标全量键值、每个 axes 的范围、每个 artist 的
数据哈希，拆前拆后逐值比对**完全一致**。协议记在
`cairn/static-gates-and-refactor.md`。

### vendored pllsim 推进到上游 `931cfaf`，C3 了结（2026-09-13）

起因是 ROADMAP C3："`np.trapezoid` 的兼容别名应当推回上游 `pll_simulator`"。
**去查证时发现前提已不成立**——上游早在 `1b0f308` 自己修了，而且修得更稳：
两侧都走 `vars(np)`，不像本仓当时的 `getattr(...) or np.trapz` 会让
`np.trapz` 成为一次静态属性读取（numpy 2 里该属性不存在）。上游还配了真的
numpy 下限 job。扫过上游 `src/` 无同类残留。所以**没有开 issue，也没有改上游
任何一行**。

C3 的后半句"这里恢复 verbatim"则顺势做掉了：`pllsim` 子树 39 个文件推进到
`931cfaf`（八个月演进，+2684/−516，19 个文件有实质变化），并补入上游新依赖的
`core/jit.py` 与 `core/boundaries.py`。`jitter.py` 的本地补丁随之删除，
manifest 从 3 条降到 2 条。

**行为完全没变，这是量出来的**：16 个 preset ＋ 3 个带损伤配置的全部指标、
每个 axes 范围、每个 artist 的数据哈希，升级前后逐值比对 19/19 一致；
272 passed / 11 skipped 不变；18 个 example 全过。

`arch/adpll.py` 与 `arch/frac.py` **没有**跟进，原因是结构性的：上游把逐周期
循环搬进了 jit kernel（numba 与纯 Python 两条路径要求逐位一致，per-cycle 路径
禁用 numpy 数组操作），而本仓的扩展每周期回调一个 Python 对象做在线两点增益
校准，装不进去。见 ROADMAP C5。

由此 pin 变成混合的，带出一个真陷阱并已堵上：sibling checkout 若是浅的，
校验器解析不到另一个 commit，会把这两个文件**跳过**——恰好是唯一两个有本地
改动的文件，而跳过原本不算失败。现在 CI 用 `fetch-depth: 0`，并给
`tools/vendor_check.py` 加了 `--fail-on-skip`（跳过即红，但仍容忍 stale pin）。
开关做了变异验证：sibling 不可达时加开关退出 1、不加仍退出 0。

### 打包元数据：classifiers / urls / py.typed（2026-09-12）

分发层此前只有名字、版本和依赖：**0 个 classifier、没有 urls、没有 `py.typed`**。
补上 11 个 classifier（逐个对着官方 trove 列表核过）、仓库/CHANGELOG/架构文档三个
`[project.urls]`、keywords，以及 PEP 561 的 `py.typed`——实测 wheel 里确实带上了
`polartx/py.typed`。

**没加 `License :: OSI Approved :: MIT License` 分类器**是有意的：PEP 639 已经
弃用 license 分类器，改用 SPDX 字符串，而字符串写法要 setuptools ≥ 77，本仓
build-system 的下限是 68（Android wheel 构建那边就是这个量级）。license 表里
已经写着 MIT，wheel 的 `License: MIT` 也正常出来了。

两处会互相说谎的地方由 `tests/test_packaging.py` 卡住：`pyproject.toml` 的版本
与 `polartx.__version__` 必须一致；`py.typed` 的标记文件和 `package-data` 条目
必须同时在——**少任何一半都是静默空操作**，装到别人机器上注解就消失了。
三个检查都做了变异验证。

版本号仍是 `0.1.0`，发布流程（tag / release / PyPI）仍未接，见 ROADMAP C1。

### Android APK：Chaquopy + WebView（2026-08-25）

第四个前端。`android/` 用 Chaquopy 把真的 CPython + numpy/scipy/matplotlib
装进 APK，UI 是 WebView，计算全在本地，**零权限**；所有逻辑经
`src/polartx/appbridge.py` 的**单函数** JSON 桥落到既有的 `guiutil`，所以手机
是同一批数字的第三个渲染器，不是第四份实现。

- 可行性闸门定在 **Python 3.10**（Chaquopy 仓库里有 scipy wheel 的最新版本），
  依据是姊妹库 `pll_simulator` 同样三个二进制依赖的 14 次实测构建。本仓代理封了
  `chaquo.com`，Gradle 构建只能在 CI 上跑。
- **编译版**：42/101 模块 Cython 编译成 `.so`。编译集不是选的是**证明**的——
  干净 venv 装 wheel、确认磁盘上没有 `.py` 可回退、跑全量：242 passed。
  顺带测掉一个真风险：`presets` 靠 `inspect.signature` 分派，Cython 化后仍可用。
- **保护边界带对照组实测**，而且第一次对照就失败（搜索串写错，当时那批阴性
  结果毫无意义）；改对后：编译模块的散文全无，**字面常量全在**（`2e6`/`50e-9`/
  `0.15` 各精确命中一次）。所以只说"把读算法的成本从解压即读抬到反汇编"。
- CI 一次出两个 APK，并用 `inspect_apk.py` 证明它们**确实不同**——两次构建共用
  工作区，真实失败模式是第二次复用第一次的 pip 输出而日志只字不提。这个门禁
  被弄坏过一次确认会红（五个方向）。
- 过程中工作区被容器回收静默退回旧提交，从 origin 恢复；重装环境后 Qt 的 GL 库
  缺失让 `test_gui_qt` 从 skip 变 ERROR，补库后**真的跑起来**：264 passed /
  10 skipped（此前 238）。
- **未做**：真机侧载。CI 绿不等于真机 import 成功。

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
