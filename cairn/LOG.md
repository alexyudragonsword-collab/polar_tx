# Project Cairn 日志

本文件按倒序记录实质性进展——最新的一条在这行下面。每条保持简短，只写摘要和指针；结论沉淀进 `cairn/<topic>.md`。

## 2026-10-02 · 记忆反演阶段 2：慢状态在合成完整源上走通，三条认识被实测改掉

- `fit_residual_memory(ch, source=)`：step 组离线辨识 τ，burst 组训练残差
  `StateConditionedSpline`；`dpa_from_complete_source`、`synthetic_thermal_source`
  （新 vendor `pa/thermal.py`、`drift.py`、`reference_pa.py`，逐字节）。
- 验收：τ 5.13 / 29.59 µs（真值 5 / 30）；burst 上状态比 SplineGMP 好 8.6 dB；预热平稳
  主采集上差 0.12 dB；缺组报错点名；入链对实测 burst −32.0 dB。故意弄坏：不预热 → 对照
  差 3.1 dB 变红；τ ×0.01 → 增益 0.4 dB 变红。
- 改掉的认识：①平稳对照要预热且够长（PA_DPD 示例主采集冷启动 54 µs，假性好 5.4 dB——
  已写进 PA_DPD `docs/02_data_interface.md` §9，PA_DPD 63cd700）；②虚拟 DUT 分块 FIR 重启
  自带 −34.9 dB 地板，阶段 2 的 NMSE 受它限制（上游问题，未改 vendored 副本，测试钉住）；
  ③状态样条按结构秩亏 49/180，条件数改报张成空间上的值，阶段 1 数字不变。
- τ 用错：×0.3～×3 不敏感，×10 太慢反而差 12 dB——要测不要猜。
- **更正阶段 0 的 LOG**：44cbcb3 所在分支 `claude/digital-polar-tx-dev-r0c338` 后来被删，现在
  从 `claude/lucid-einstein-58n326` 可达，仍不在 main；文档改为"分支名会变、SHA 不会"。
- 分支 `feat/slow-memory`，PR #6。笔记 `cairn/measured-memory.md` 新增阶段 2 一节。

## 2026-10-01 · 记忆反演阶段 1：残差 GMP 进 chain.memory，"19 dB 是记忆"经住了对照

- `measured.py`：`static_prediction` / `fit_residual_memory` / `load_measured_dpa(with_memory)`，
  训练对 (静态输出, 实测输出) 除以 `chain_gain` 到链路尺度，前 80% 训后 20% 验。
  极坐标链路 `run()` 一行未动，`chain.py` 只改 `memory` 的 docstring。
- 数字（验证集）：160 MHz 静态 −19.95 / 无记忆残差 −19.97 / GMP-510 −37.5 / SplineGMP
  −37.7（直拟 −38.7，发表 −39.2）；200 MHz −20.54 / −20.56 / −33.3（发表 −33.7）。
  无记忆对照只买 0.02 dB → **19 dB 全是记忆**，不是分箱静态模型抹平了形状。
- 入链（峰值放到实测满量程、同采样率）：极坐标 DPD 静态 −32.0→−49.9，有记忆
  −19.75→−19.94；整链 ILA（GMP-510 结构）−19.75→−36.5，默认结构只 13 dB；
  ILA 让 ACLR 变差 4 dB，如实记。npz 往返逐位相同。链路+残差对实测 −37.8。
- 两个口径坑进 `cairn/measured-memory.md`：尺度必须等于链路 y 的尺度；对齐只做
  整数延迟（小数重采样让合成 DUT 复合从 −52 掉到 −46.6 dB）。
- 与方案出入：SplineGMP 条件数低 1.4 个量级而非 2–3；静态 64 箱地板 −35 dB 来自
  顶部稀疏箱钳位、与箱数无关（留待后定项，未动）。ROADMAP B2 改写。
- 分支 `feat/measured-memory`，两个 commit（阶段 0 / 阶段 1），PR #5。

## 2026-10-01 · 记忆反演阶段 0：padpd 子树重拷到 44cbcb3，差异归因为"全是上游前进"

- 疑问是 `pa/{base,gmp,memory_polynomial}.py` 与上游主干差 42/15/20 行是谁动的。
  全克隆 PA_DPD 后逐字节对照钉定 blob 44f9ee99：三个文件（以及其余 8 个 stale 文件）
  的 vendored 副本与钉定 blob **零差异**，本地从未改过——不需要回游，也不进 manifest。
- 上游变化内容：`lstsq_fit` 加 keyword-only 的 weights / penalty（默认同旧行为）、
  `basis_cond`、MP 的 `n_coeffs`/`gain_curve`、ILA 跳过末轮无用的 `model(x)`、
  psd 加 −200 dB 地板、hb_import 改用 `data/io.read_csv_columns`。重拷全部 19 个
  padpd 文件到 44cbcb3，新 vendor `pa/spline.py`、`pa/spline_state.py`、
  `gain_modulation.py`、`data/complete.py`，抽取 `pa/presets.py`（manifest 记 hash）。
- 验证走 dump-比对协议（`static-gates-and-refactor.md`）：ILA 链 EVM −20.12→−68.64、
  ACLR −27.21→−52.99、OpenDPD 160/200 静态 NMSE −19.93/−20.60、PSD/CFR/对齐输出
  哈希——重拷前后 JSON 逐字节相同。test_memory_dpd / test_measured / test_vendor_* 全绿。
- 校验器新增 `pin_frozen`：pllsim 的 adpll/frac 两条 pin 是**故意**停在 d7be4712
  （C5），以前 `--strict` 永远红；现在报 *frozen* 并带原因，`--strict` 可过而信息不丢。
  `__init__.py` 校验器不跟踪（裁剪版包装），`pa/__init__` 的注册表扩到全部 vendored 模型。
- 跨库往返：PA_DPD 侧写的 GMP npz 进 `tests/data/`，vendored `load_model` 读回后输出一致到
  浮点精度（BLAS 求和顺序在 macOS / 下限 job 差 1 ulp，`array_equal` 被 CI 抓到）。基线 323 passed。
- 坑：44cbcb3 是 PA_DPD `claude/digital-polar-tx-dev-r0c338` 的头，**不在其 main**（3aa1c24，
  落后 21 个提交）；CI 的 sibling checkout 要钉全 sha + `fetch-depth: 0`。

## 2026-09-30 · B7 收口：outphasing 两路共用一个 LO，3 dB 差距整个消失

- 修法绕开 `chain.py`：`DTCPMConfig.lo_pn_seed`（None 逐位不变；整数 = LO 样本走
  `default_rng([seed, 1])` 独立流），`OutphasingTX.run` 每次浅拷贝分支链钉住本次 seed。
  抖动/dither 仍独立；Ideal/ADPLL 无 LO 模型不受影响。改了 `phasemod/` 一个字段
  三行逻辑——用户看过 B7 说"继续"，视为放行；`chain.py` 仍未动。
- **更正**：阶段 1 说 outphasing 差 3 dB 是"两分支满功率、相噪不相消"，阶段 3 说
  其中约 3 dB 是独立 LO。实测是**全部**：共用 LO 后 −39.84 vs −40.00（地板差 0.4 dB，
  4 符号 −0.3），与选型器共模预算 0.44 一致；独立 LO 时 5.1 / 4.0 dB。原因：这条链
  地板由 LO 主导（−49），DTC 量化/抖动（−67/−55）各多付 5.5 dB 也不露头。
- 教训：分支级噪声"独立"要逐个来源问一遍，共享的硬件块（LO）不能跟着运行 seed 走。
- 数字变动：ACLR −49 → −53 dBc，分支相位 1 dB 容忍度 0.74° → 0.51°；ex19/ex20/README
  同步。ROADMAP B7 删除。分支 `feat/outphasing-shared-lo`，PR #4。

## 2026-09-30 · 阶段 3：选型器四候选，两个建模缺口被对照测试揪出

- `selector.py` 四候选：outphasing = DTC 地板逐项 +10log10(PAPR/2)（分支恒满幅）
  + 失配闭式（带内占比 −7.5 dB 为链路实测常数）；RF-DAC = 两轴量化/OSR + 抖动
  + 镜像 −IRR。效率各按自己的律在截断瑞利分布上积分，与链路差 < 0.01。
- **决定**（方案文档"阶段 3 前得定"）：排序 = 可行 → 达标 → 达标者中效率最高 →
  EVM 最低；`eta_avg_min` 为硬门槛。达标后的 EVM 裕度不值钱、效率值钱。
  验收：BLE 仍 ADPLL；320 MHz 4096-QAM 不选 outphasing（差 7.3 dB、效率更低）。
- **更正阶段 2**：RF-DAC 链没有 LO 相噪，同台表 −42.4 dB 比选型器乐观 15 dB。
  补 `RFDACConfig.lo_pn/lo_loop_bw`（同 DTC 字段与生成器），预设带同一 LO；
  RF-DAC 改为 −40.0 dB，与极坐标持平——两者同被 LO 限住。抽象没变，数字变了。
- **更正阶段 1 的认识**：outphasing 链两路 LO 相噪独立抽（seed/seed+1），真机共模；
  链路里 outphasing 比极坐标差的 ~5 dB 中约 3 dB 由此而来。修它要碰主链，记
  ROADMAP B7；选型器默认共模（`outphasing_shared_lo`），对照测试显式用 False。
- 一个模型边界：截断瑞利在 PAPR 3.4 dB（8DPSK）时削顶质量 11%，自身 PAPR 3.9 dB。
- ex20 加"最敏感旋钮 1 dB 容忍度"列：skew 0.03 ns / 分支相位 0.74° / I/Q 相位 0.62°。
  三前端经 `run_selector_report` 同步拿四行 + 效率 + 四曲线，`appbridge.py` 未动。
- 分支 `feat/selector-three-topologies`（叠在 `feat/rfdac-tx` 上），PR #3，base 先指
  #2 的分支，#2 合并后自动转 main。本地 317 passed / 11 skipped（+README 计数已修）。

## 2026-09-30 · RF-DAC 拓扑阶段 2：第三种架构进同一张表

- 按方案文档做：`rfdac.py`（I/Q 两阵列，失配复用 `dpa/mismatch.py`，I/Q 失衡 /
  LO 泄漏 / 抖动，`iq_cells` 效率律）+ `cartesian.py`（`CartesianTX` 吃同一个
  `ChainConfig` 只读 `cfr_papr_db`，其余字段非默认即 `UserWarning` 点名）。
  `CartesianResult` 自实现同名指标——这是第三个结果类，仍没有 isinstance 分派。
  `chain.py` / `ChainConfig` / `PolarResult` 未改一行。
- 五条验收成测试并有实测值：理想链 EVM 差 0.001 dB；单音镜像 = 解析 IRR 39.61 dB
  （精确），调制 EVM 退化贴 −IRR；I 阵列 INL/DNL 与 `DPA.inl_dnl()` 逐元素相等；
  6 dB 回退 42.5% < SCPA 50.9%、burst 平均 28.3% < 42.8%；`env_skew_s` 告警且逐位相同。
- 一个断言前提被实测推翻：10 bit 6+4 分段阵列 1% 失配只让误差到 −72 dB，比 CFR
  地板低 30 dB，EVM 不动。改为断言误差功率随 σ_cell 20 dB/十倍（实测 20.0），
  EVM 不变反而是分段的物理结论。另：4× 过采样把 3/4 量化噪声推到带外，8 bit
  只差 0.3 dB，分辨率测试用 6 bit（差 3 dB）。
- ex20 同台表（加噪）：极坐标 −40.0 / −58 / 42.8%；outphasing −37.1 / −49 / 40.0%；
  RF-DAC −42.4 / −60 / 28.0%。线性最好效率最差，教科书取舍成立。
- 判断：抽象第三次站住。下一步抽 `TXResult` Protocol 的信号仍是报告层出现
  isinstance，`ex20` 里的 `hasattr(res, "rfdac")` 是目前唯一一处分派，留待后定。
- 分支 `feat/rfdac-tx`，PR #2，CI 全绿（含 vendor-drift）。本地 303 passed / 11 skipped。

## 2026-09-30 · Outphasing 拓扑阶段 1：抽象站住了

- 按方案文档做：兄弟类 `OutphasingTX` 包住 `PolarTX` 跑两路恒包络分支，
  `OutphasingCombiner`（isolated / Chireix）合路，预设 `.tx` 别名进注册表。
  `chain.py` / `ChainConfig` / `PolarResult` 未改一行——抽象的核心检验。
- 五条验收全部成测试并有实测值：理想链 EVM 差 0.000 dB；1° 相位失配退化 vs 闭式
  预算差 0.35 dB 且单调；isolated 效率 = η_pa·E[cos²θ]（1e-9）、Chireix 43.9% >
  isolated 10.9%；θ>80° 占比 21%；`env_skew_s` 逐位无效而极坐标退化 20 dB。
- 同台表（ex19，真实 DTC + Rapp DPA）：EVM −37.1 vs −40.0，ACLR −49 vs −58，
  效率 Chireix 40.0% / 极坐标 42.8%。3 dB 差的物理：两分支满功率、相噪不随包络
  回退且两路独立不相消——这是拓扑本身的代价，不是实现 bug。
- 判断：抽象站得住。"兄弟类 + 同名指标方法 + 预设别名"两次（FIR、outphasing）
  都没碰主链；下一步该抽 `TXResult` Protocol 的信号是报告层出现 isinstance，
  目前没有。已知取舍：Chireix 负载调制只进效率律。
- 容器回收后环境被清空（pytest、Qt GL 库），`apt-get update` 后补装即可；
  main 基线在此环境下是 265 passed / 7 errors 而非 272，全是 `libEGL` 缺失。
- 分支 `feat/outphasing-tx`，PR #1。CI 11/12 绿；`vendor-drift` 红的根因是
  `pll_simulator` 已转 private，Actions 默认 token 取不到（`PA_DPD` 同 job 正常），
  `--fail-on-skip` 正确拒绝了 22 个不可校验文件——与本 PR 无关，main 同样会红。
  已加 `SIBLING_READ_TOKEN || github.token` 回退，需仓库 secret 或改回 public。

## 2026-09-13 · 查证 C3：上游已自行修复，顺势把 pllsim 子树推进到 931cfaf

- 任务是"把 `np.trapezoid` 的修法推回上游、由上游 agent 决定"。**查证后没有上报**：
  上游 `1b0f308` 早已自修，写法（两侧 `vars(np)`）比本仓更稳，还配了真的 numpy
  下限 job；扫过上游 `src/` 无同类残留。ROADMAP C3 的前提已不成立，按事实重写。
- 顺势完成 C3 后半句：`pllsim` 子树 39 个文件推进到 `931cfaf`（+2684/−516），
  补入上游新依赖 `core/jit.py`、`core/boundaries.py`；`jitter.py` 的本地补丁删除，
  manifest 3 条 → 2 条。**19 个配置逐值比对完全一致**，272 passed 不变。
- `adpll.py` / `frac.py` 没跟进，是结构性原因：上游把逐周期循环搬进 jit kernel
  （两条路径要求逐位一致、per-cycle 禁 numpy 数组操作），而本仓扩展每周期回调
  Python 对象做在线两点校准。记为 ROADMAP C5，不是待办漏项。
- 由此带出的真陷阱已堵：混合 pin + 浅 checkout 会让校验器**跳过**那两个唯一有
  本地改动的文件，而跳过原本不算失败。CI 改 `fetch-depth: 0` 并加
  `--fail-on-skip`（变异验证过）。细节见 `cairn/static-gates-and-refactor.md`。
- Android 编译集 42/101 是 2026-08-25 的实测，分母现已 103，**分子未重测**——
  历史记录一律不改，另立 ROADMAP C6。

## 2026-09-12 · 工程体检后的四批整改：静态闸门、类型、文档锚定、拆报告层、打包

- 体检（只读）结论：主干健康——277 项测试 0 失败、自身覆盖率 92.8%、vendor 0
  漂移、bandit 无中高危；短板全在工程外围。决策与教训见
  `cairn/static-gates-and-refactor.md`。
- **1a** ruff 进 CI（版本钉死），抓出 3 处真引用丢失 + 18 处死导入。
- **1b** mypy 进 CI。根因是 `ofdm_ref` / `fir_tx` 用 `object` 当占位，让穿过它的
  一切都不可检查；顺出 `seed=None` 下带 pilot 的波形死在算术里等真问题。
- **2** `docs/architecture.md` 补到文件级（14 个模块此前只有目录级概括），
  新增 `tests/test_docs_consistency.py` 两方向卡死，README 计数改为可核对的数。
- **3** 拆 `run_chain_report`（168 行 D(29) → A(3)），计算与绘图分离。验证不靠
  "套件还绿"：19 个配置的指标与图上每个 artist 逐值比对完全一致（协议见专题文档）。
- **4** 打包元数据：11 个 classifier（对着 trove 列表核过）、`[project.urls]`、
  PEP 561 `py.typed`（实测在 wheel 里）；版本两处一致与 py.typed 的"标记 +
  package-data 缺一不可"由 `tests/test_packaging.py` 卡死。ROADMAP C1 改写未删。
- **修订**：1b 声称的"mypy clean"是假的——本地 mypy 装在 uv 独立环境里看不见
  numpy/PySide6，把一切退化成 `Any`；CI 同环境跑出 9 个错（8 个真该修）。已在
  专题文档教训区就地更正并写明堵法。
- 未做（需授权）：合回 `main`；tag / release / PyPI（ROADMAP C1 剩余、C2）。

## 2026-08-25 · Android 面：Chaquopy + WebView，解释版与编译版

- 按 `python-android-apk` 技能全流程做完：可行性闸门 → appbridge → android/ 骨架
  → Cython 编译变体 → CI。详见 `cairn/android-app.md` 与 `docs/android.md`。
- 闸门结论 Python 3.10（scipy 的 Android wheel 到此为止），依据是姊妹库
  `pll_simulator` 同样三个二进制依赖的 14 次实测构建，不是推断——本仓代理封了
  `chaquo.com`，所以本地跑不了 Gradle，那是 CI 的事。
- 编译集 **42/101 模块**，用"删掉 .py 后套件仍过"证明：干净 venv 装 wheel、
  确认无 `.py` 可回退、**242 passed / 12 skipped**。
- 编译到底买到什么：**带对照组**测的，且第一次对照就失败（搜索串写错），
  改对后才成立——散文全无，**字面常量全在**。
- 工作区中途被容器回收静默退回 `22b7519`，从 origin `reset --hard` 恢复；
  Python 环境同时被清空，重装后 Qt 的 GL 库缺失使 `test_gui_qt` 从 skip 变
  ERROR，补装 `libegl1` 等后**真的跑起来**：全量 **264 passed / 10 skipped**。
- **真机侧载做了，暴露出一个真 bug**：构建全绿、APK 能装，但**运行按钮全无反应**
  ——`applyLang()` 的 `textContent` 把 21 个 `<label data-zh=…>` 包着的控件全删了。
  纯文本 parity 测试结构上看不见这类"运行时被销毁"。补了静态不变量 + jsdom
  真 DOM harness（CI 独立一步直接跑 node，不走会 skip 的包装）。详见
  `cairn/android-app.md`。
- **首次真构建：解释版成功（83.9 MB artifact），编译版挂在 `cpow` 未声明**。
  修在工具链（`-include complex.h` / `-lm` / 仅交叉路径的 `-Wl,--no-undefined`），
  不是移除模块——扫过生成 C 才发现 13/42 个模块都会调 `cpow`。详见
  `cairn/android-app.md`。
- **第二次构建否掉了那个修法**：带着 `-include complex.h` 仍报同一条错——
  **Bionic 在该 API 级别没有 `cpow` 这个函数**。真开关是 `-DCYTHON_CCOMPLEX=0`
  （Cython 自带复数实现，只用实数 libm）。上一次的"对照验证"做在了 glibc 上，
  而缺的从来不是 glibc 的声明——交叉编译的验证必须在目标侧约束下做。
- **同一版还把 CI 弄红了，我当时没看**：`test_android_parity.py` 裸 `read_text()`
  走 locale 编码，Windows runner 的 cp1252 撞上页面里的中英双语串，14 项挂
  `UnicodeDecodeError`（10 个 job 只红这 1 个）。已收口到统一的 UTF-8 helper。

## 2026-08-18 · Project Cairn 初始化

- 在已成熟的项目上做 retrofit：`polartx` 此时已有 238 项测试全绿、18 个 examples、两个 GUI、40 次提交。
- 配置：`git_policy: track`（本仓文档全公开，知识层同样公开最一致）、provider **暂缓对接**、`language: zh`、`migration_mode: inventory_only`。
- 原有的 `CLAUDE.md` 不是丢弃而是**升格**：内容整体搬进 `AGENTS.md` 并补上 Cairn 的规则层，`CLAUDE.md` 退化成一行 `@AGENTS.md` 存根。差异见本条下面的"初始化对 CLAUDE.md 的影响"。
- 本项目**不建** `cairn/ROADMAP.md`：根目录已有 `ROADMAP.md`（只写未完成项），再建一份会重演"两个文件抢同一个名字"的老问题。AGENTS.md 的阅读顺序里直接指向根目录那份。
- 历史清单（`inventory_only` 的产物）：见 `cairn/history-inventory.md`。
- 细节见 `AGENTS.md` 与 `.cairn/config.yaml`。
