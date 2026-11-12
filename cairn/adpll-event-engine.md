---
type: project_topic
status: active
summary: "ADPLL 两点调制的 event（逐参考周期）引擎：B1 的真因是 LTE-20 预设的 TDC 量程只有 DCO 周期的 1/4，不是'相位推进 ≪1 UI'；修好后剩下的 event/response 之差来自 TDC 按标称周期归一化；vendored adpll.py 钉在旧版让上游的现成提示没到我们手里"
tags: [polartx, adpll, event 引擎, tdc, 两点调制, vendor, fmcw_radar]
contains: [decision, pitfall, lesson, experience]
created: "2026-11-12"
updated: "2026-11-12（C5 了结）"
related: [history-inventory.md, static-gates-and-refactor.md]
authoring_mode: ai_generated
---
# ADPLL event 引擎

## B1 的真因（pitfall，2026-11-12 定位）

2026-08-10 起，`ADPLLTwoPoint(mode="event")` 被声明为"只在每参考周期相位推进 ≪1 UI 时
有效"：BLE 0.8 %、EDR 4.2 % 可以，LTE-20 29 % 读 −15 dB（response −37 dB），于是超过
10 % 就告警，ROADMAP B1 记成"event 引擎覆盖不到 LTE 级带宽，真因未定位"。

触发重新看的是姊妹库 fmcw_radar（polar_tx#19）：1 GHz 锯齿扫频每周期推进 495 %，
event 与 response 却逐周期一致（飞回瞬态误差比 1.005 ± 0.046）。推进量 17 倍于 LTE-20
却没事，说明推进量不是那个变量。

真因：**LTE-20 预设写的是 `TDCConfig(t_res=1e-12)`，n_bits 落在默认 7，量程 127 ps，
而 1.95 GHz 的 DCO 周期是 513 ps**。计数器辅助 ADPLL 用计数器数整周期、只用 TDC 读
小数；小数超过 0.25 UI 的周期 TDC 饱和，环路对那部分周期是瞎的。只改 n_bits（同一 LTE
波形，相位误差 RMS，噪声开）：

| n_bits | 量程/周期 | event | response | 不调制锁定 |
|---|---|---|---|---|
| 7 | 0.25 | −13.5 dB | −52.7 dB | 0.109 rad |
| 8 | 0.50 | −17.0 dB | −52.7 dB | 0.071 rad |
| 9 | 1.00 | −44.2 dB | −52.7 dB | 0.0013 rad |

链路层面（`run_chain_report("LTE 20 MHz", mode="event")`，seed 1）：EVM −14.7 → −44.5 dB，
ACLR −24.0 → −60.1 dBc，mask FAIL → PASS；response 引擎逐项不变（−53.3 dB）。BLE 的
10 ps × 127 = 1.27 ns、EDGE 的 20 ps × 127、fmcw_radar 的 5 ps × 127（15 GHz）都覆盖
一个周期，所以只有 LTE 坏了——"推进量大的那个坏了"是巧合。

**决策**：预设按 `fout` 算位数（`presets._tdc_bits_for_period`）；`ADPLLTwoPoint` 构造时
检查量程 ≥ 1 个 DCO 周期，不满足抛 `ValueError`（两种引擎都查：response 的噪声预算描述的
也是一个能用的 TDC）。按推进量的告警删除，`ui_per_ref_cycle_p99` 只作描述留在诊断里。
测试：`test_the_tdc_must_span_a_dco_period`、`test_event_engine_has_no_phase_advance_limit`
（LTE-20 误差须低于归一化上限 2π·rms(Δf)/f；旧 TDC 下是该上限的 8 倍）。

## 剩下的 ~9 dB 是一项物理，不是 bug（decision）

TDC 补够后 event 仍比 response 差约 9 dB（相位误差 −44 vs −53 dB），并随调制幅度增大
（×0.1 / ×1 / ×3 → 0.0032 / 0.0065 / 0.0173 rad），与 TDC LSB、DCO 步长、DCO dither 无关。
来源：vendored `_sim_tdc` 用 `tdc_ui = code / cpp_nom`，即按**标称**周期把码值换成 UI；
LTE 调制让 DCO 偏离 ±2 %（`dco_dev_frac_p99` ≈ 0.018），小数相位因此读错 `frac·Δf/f`。
临时把这一行换成按实际周期归一化，event 噪声开 −52.6 dB，与 response −52.7 dB 一致到 0.1 dB。

**保留现状**：真实 ADPLL 也是按平均周期归一化 TDC，所以这是该被建模的效应，缺的是线性
模型里这一项。BLE/EDR 的 Δf/f ~1e-4，两引擎一致到 0.1 dB。新增诊断 `dco_dev_frac_p99`
给出决定这一项大小的量。

## 证伪过的假说（experience）

- 2026-08：两点通路"差一拍"——两个方向移位都更糟。
- 2026-11-12：亚周期时间对齐差（误差 ∝ 瞬时频偏 × δt）——按 dφ 回归拟合出 τ = 0.015 周期，
  残差不降。两个假说都在找"推进量大会出什么错"，问错了问题。

## 教训（lesson）

- **适用域要从机理推，不要从一个失败的例子归纳**。当年的规则只问了"哪个预设坏了"，没问
  "为什么坏"；三个预设里恰好推进量最大的那个配置有错。
- **钉着旧版 vendor 的代价会在别处出现**。上游 pll_simulator 在 v0.9.1（`3bd73cb`，2026-08-10，
  正好是 B1 写进来那天）给 `analyze()` 加了"TDC range is 0.25 of an output period: it saturates
  every cycle"——对我们的 LTE-20 参数实测能报出来。`adpll.py` 因 ROADMAP C5 钉在 `d7be4712`
  （07-23），这条提示三个月没到我们手里。
- **姊妹库的反例比自查更有效**：同一个引擎在别人的参数下没坏，是唯一能把"引擎限制"和
  "配置错误"分开的证据。

## dp_cal 在频率斜坡上（fmcw_radar 报告，未在本仓复现）

在线 `SignSignLMS` 用 EMA 去均值的调制频率作回归量，要求零均值调制。锯齿扫频的回归量
几乎恒正，失配只在每次飞回后约 25 个周期可见，fmcw_radar 实测 +5 % 失配 20 个 chirp 后
1.05 → 1.066（发散）；离线 `estimate_dp_gain_error` 喂 ≥2 个 chirp 一次估到 4.995 %。
已写进 `ADPLLTwoPoint.dp_cal` 与 `cal/dp_cal_fixed.py` 的说明，不改算法。

## 外部消费者

fmcw_radar 经本仓的 vendored `adpll.py` 用这套环路，登记在等 C5（上游 #77 TDC 抖动项与
死区提示、#78 抖动回绕），不阻塞：它自己在剖面里补了同一抖动项，只在分项缺 `tdc_jitter`
时补。本仓 ADPLL 预设不设 `tdc.jitter_rms_s`、FCW 都是小数，跟进后不动任何数。

## C5 了结：引擎换成上游的内核（2026-11-12，decision）

`adpll.py` 从 `d7be4712` 推到上游 `5564c64`，整棵 `pllsim` 子树单一 pin。`dp_cal` 不再
是每周期回调的 Python 对象，而是 TDC 内核里的一个步进函数加状态向量（上游 TDC 周期 /
Kdco 校准器的同一种做法）；旧式只有 `.step/.value` 的对象会被按名拒绝。量程检查改用
上游的 `core/tdcspurs.code_span`，上游 `analyze()` 的"TDC range is 0.25 of an output
period"提示现在也到手了（测试里断言）。

本仓所有 ADPLL 路径与改前逐位一致（38/38 项指纹），所以上面所有数字不动；numba 下编译
与解释也逐位一致（CI `test-jit`）。速度（BLE event，104k 周期，`simulate()` 含后处理）：
旧循环 0.41 s，新内核解释执行 0.36 s，numba 0.07 s。
