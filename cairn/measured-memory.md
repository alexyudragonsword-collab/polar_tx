---
type: project_topic
status: active
summary: "实测 DPA 的记忆效应怎么进极坐标链路：残差模型挂在 chain.memory 上，尺度必须等于链路 y 的尺度，对齐只做整数延迟；快记忆的结论（19 dB 确是记忆）与慢状态的路径（τ 辨识 + 状态样条、平稳对照必须预热、虚拟 DUT 的 −35 dB 地板）"
tags: [polartx, 实测数据, 记忆效应, OpenDPD, 残差模型, vendor]
contains: [decision, lesson, pitfall, experience]
created: "2026-10-01"
updated: "2026-10-02（新增阶段 2 慢状态：决策、两个坑、两条经验）"
related: [static-gates-and-refactor.md]
authoring_mode: ai_generated
---
# 实测记忆的反演与入链

代码在 `src/polartx/measured.py`（`static_prediction` / `fit_residual_memory` /
`load_measured_dpa(with_memory=True)`），数字在 `examples/ex09`。这里只放过程知识。

## 决策：残差模型，不是替换

记忆模型不替代 DPA，而是训练在 `(静态模型输出, 实测输出)` 上的残差 `PAModel`，
挂在 `PolarTX(memory=...)`——`run()` 里 DPA 合路、乘回 `fs_scale` 之后那一级。
静态部分留给 DPA 的 LUT（极坐标 DPD 能修的），记忆部分留给残差（只有笛卡尔
ILA 能修的），两段各归各的物理。模型以 padpd 的 npz 形式（`PAModel.save` /
vendored `load_model`）进出，读回后入链结果逐位相同。

## 坑：残差模型的尺度必须等于链路 y 的尺度（pitfall）

链路名义增益为 1：`env_cmd = env / fs_scale`，输出 `y = dpa(...) * fs_scale`。
实测对 `(x, y)` 却带着器件增益（OpenDPD 160 MHz 约 2.9，LUT 顶箱口径 2.24）。
残差模型的 |y|^k 项**不是尺度不变的**：在实测伏特尺度上拟、在链路尺度上跑，
所有非线性项全错。所以训练对是 `static_prediction(ch, x) / chain_gain` 对
`y_aligned / chain_gain`，其中 `chain_gain = r_in[-1] * g_bin[-1]` 正是 LUT
归一化除掉的那个数；调用方必须 `ChainConfig(fs_scale_fixed=ch["fs_scale"])`
（实测 x 的满量程）并把波形峰值放到同一满量程，采样率等于 `ch["fs"]`
（抽头以采样计）。验证：链路静态输出对训练输入 u 的 NMSE −54 dB（10 bit 量化），
链路 + 残差对实测输出 −37.8 dB，与残差模型自己的验证集 −37.5 dB 一致。
`cal/memory_dpd.py` 要求 `fs_scale_fixed` 的原因是同一个。

## 坑：对齐只做整数延迟（pitfall）

vendored `align_delay` 会把 > 0.02 采样的小数延迟用 FFT 相位斜坡重采样掉。
对后面接记忆模型的流水线这是错的：小数延迟就是线性记忆（群延迟），残差模型的
抽头正该表示它；先重采样等于塞进一个 sinc 插值器让模型去抵消。合成 DUT
（Rapp + FIR [1, 0.12+0.04j] + 0.05·y|y|²）上：小数重采样后复合 NMSE −46.6 dB，
只做整数对齐 −52 dB，且残差线性抽头与注入 FIR 差 < 3%。于是 `measured.py`
默认 `fractional_align=False`；小数延迟仍估计并报在 `align["lag_total"]`。
代价：OpenDPD 160 MHz 的静态 NMSE 从 −19.93 变成 −19.95（它的 lag 是 −0.031）。

## 经验：64 箱 LUT 的静态地板 ≈ −35 dB，来源不是箱数

合成纯静态 Rapp 器件上 64 / 128 / 256 箱的静态 NMSE 都是 −35 dB——
来源是**顶部稀疏箱被丢弃（< 8 样本）后插值被钳在最后一个有效箱**，
峰值样本吃到错的增益。残差 GMP 的多项式项把这段平滑失配一起修掉了
（复合 −52 dB），所以阶段 1 没动静态提取；要把静态模型本身做到 −50 dB
得换样条（方案文档"留待后定"项）。

## 结论："19 dB 就是器件记忆"经得起无记忆对照

在 LUT 之上再加一个 5 阶静态多项式（`memoryless_residual_factory`），验证集
NMSE 只从 −19.95 到 −19.97（160 MHz）、−20.54 到 −20.56（200 MHz）：
**0.02 dB**。静态模型没有漏掉任何静态形状；差距全部要记忆项才补得上
（GMP-510 残差 −37.5 / −33.3，PA_DPD 直拟 −38.7 / −33.8，发表 −39.2 / −33.7）。
入链后极坐标 DPD 在静态器件上买 18 dB（−32 → −50），记忆进链后只买 0.2 dB
（−19.75 → −19.94）；整链 ILA 买回 13 dB。

## 经验：SplineGMP 在残差问题上条件数只低 1.4 个量级

PA_DPD 在 x→y 直拟上报告样条基比多项式基条件数低 2–3 个量级。残差问题的输入
已经是压缩后的包络，多项式基的共线性没那么糟：GMP-510 条件数 2.1e5，同为
255 系数的 SplineGMP（3 节点、三次、分位数放置）8.8e3——低 1.4 个量级，
NMSE 好 0.2 dB。测试按实测卡 ≥ 1 个量级，不按方案里的 2。

## 阶段 2：慢状态（2026-10-02）

**决策**：慢状态和快记忆走同一条残差路径，只换训练数据和模型。`step` 组离线辨识
τ（vendored `identify_gain_modulation_capture`），`burst` 组过静态模型后训练残差
`StateConditionedSpline`，状态的 α = exp(−1/(τ·fs)) 直接取辨识值。split 默认 0.6
而不是 0.8：四个 burst 时验证尾段要同时含加热和冷却沿（0.6 留 3.2 段，0.8 只 1.6 段）。
入链时每次 `run()` 是冷启动——状态模型每次调用从零重算状态，正对应 burst 采集里冷启动的
DUT。τ 是 AM→AM / AM→PM 的增益调制常数，**不是** `SupplyConfig.tau_s`，不能填过去。

**坑：平稳采集必须预热且够长（pitfall）**。"状态模型在平稳采集上不该变好"这条对照，
只有主采集真的处于热稳态才成立。PA_DPD 示例的主采集是冷启动 54 µs，那是一段加热暂态：
在它上面状态模型好出 5.4 dB（冷启动 218 µs 也还有 3.1 dB）。预热两遍、218 µs（≈7 倍
最慢 τ）后差 0.12 dB。短而热的采集（54 µs）反过来让 180 系数的状态模型**差** 3.7 dB
（训练样本不够）。所以合成源的主采集是"预热 + 长"，并写进了 PA_DPD 的数据接口文档。

**经验：τ 用错不是中性的**。在 burst 上把辨识值整体缩放：×0.3～×3 状态增益都在 8.3～8.6 dB，
×0.1 掉到 6.5、×0.03 只剩 1.8；×10（太慢）**反而比无状态差 12 dB**——状态几乎不动，
样条在一个窄区间上过拟合。这就是 τ 要测、不能猜的定量理由。

**坑：虚拟 DUT 自带 −35 dB 地板（pitfall，上游问题）**。`ThermalReferencePA` 按 128 采样
分块推进热状态，每块单独调用 `ReferencePA`，而后者每次调用都从零做 FIR 卷积——输入 3 抽头、
输出 2 抽头的 FIR 在每块开头重启，块内前三个采样出错（−14 / −25 / −50 dB），其余为零。
冻结状态（`heat_gain=0`）同输入对比一次连续调用：NMSE −34.9 dB。这是与采样序号绑定的
非物理误差，任何因果模型都拟不掉：平稳主采集上的残差（−34.0）正落在这条地板上，所以阶段 2
的 NMSE 都受 DUT 限制，模型间的增益（8.6 dB）是下界。PA_DPD 自己的 +8.5 / +10.2 dB 也同样
受影响。vendored 副本不改（vendor 规矩）；修法应在上游：FIR 跨块保持状态，或先对整段做
FIR、只把静态非线性分块。测试 `test_the_virtual_dut_floors_every_model...` 把这条地板钉住。

**经验：`StateConditionedSpline` 的原始条件数没有意义**。它**按结构就秩亏**：每个加法状态块
`x[n−m]·C_i(q)` 对 i 求和等于 `x[n−m]`（单位分解），基线抽头已经张成这个方向；交互曲面同样
复写 tap-0 列。上游靠 1e-9 的岭项取最小范数解（SplineGMP 的做法是每个交叉支路删一列）。
再加上所有状态共用一个 `q_scale`（取最快状态的峰值），最慢状态够不着顶部节点，留下全零列。
合成 burst 上 180 系数秩亏 49（结构性 28、全零 14、其余数据相关），原始条件数 ~1e18；
报告里改报"张成空间上的条件数"（1.6e4）并同时报秩亏与全零列数。满秩模型上这个数与
`basis_cond` 完全相同，阶段 1 的数字不变（160 MHz 2.1e5、200 MHz 1.47e5）。
