# 贡献指南 / Contributing

本文件把这个工程**已经在测试里强制执行**的约定写下来。它不是愿望清单：
下面每一条规矩都对应至少一个会失败的测试或一个 CI job。改动如果违反其中
任何一条，CI 会红，而不是靠 review 抓。

This file documents the conventions this project **already enforces in
tests**. Every rule below is backed by a failing test or a CI job, not by
reviewer goodwill.

---

## 0. 环境 / Setup

```bash
pip install -e ".[test,gui,guiqt]"
pytest tests/ -q                    # ~240 项，~2 min；全绿是提交的前置条件
python examples/exNN_*.py           # 图写到 examples/out/
streamlit run gui/Home.py           # 网页 GUI
polartx-gui                         # 桌面 GUI（或 python -m polartx.guiqt）
python tools/vendor_check.py        # vendor 漂移检查（CI 也跑这个）
```

可选依赖：`iverilog`（RTL 金向量验证）、`streamlit` / `PySide6`（两个 GUI）。
缺失时相关测试**干净地 skip**——所以**本地带 skip 的全绿是正常的，CI 上带
skip 就不正常**：那说明某个 job 的依赖没装上，等于那部分没被验证。

---

## 1. 测试写"物理量"，不写"快照" / Assert physics, not snapshots

**规矩**：一个测试要么断言一个可以独立推导出来的物理量，要么断言两条
独立实现的结果一致。不要把当前输出的数值 hardcode 成"期望值"——那种测试
只会在重构时报警，不会在做错时报警。

已在库里的例子，可以照抄这些形状：

| 形状 | 例子 | 断言的是什么 |
| --- | --- | --- |
| 闭式公式 | `test_dtc_pm.py` | B bit 相位量化 EVM 底 = `(2π/2^B)/√12`，±2 dB |
| 标度律 | `test_dpa_tables.py` | 单元失配 σ → INL 的 √N 律 |
| 双引擎一致 | `test_adpll_tp.py` | response（z 域）与 event（逐周期）EVM 吻合 |
| 逐位回归 | `test_ofdm_general.py` | WiFi 预设与 padpd 生成器同种子逐位一致 |
| 三方恒等 | `test_rtl_export.py` | Verilog ≡ 整数金向量 ≡ 浮点引擎 |
| 单调性 | `test_sem.py` | skew 增大 → OOB 裕量单调下降 |
| 不变量 | `test_benchmarks.py` | 每个 `bench_*` 都能从 GUI 到达 |

**反例**（不要写）：

```python
assert res.evm().db == pytest.approx(-38.27, abs=0.01)   # ✗ 快照
assert len(PRESETS) == 14                                # ✗ 硬编码计数
```

后者曾经真的挡住过一次合法的新增预设。现在写成不变量：

```python
exported = {n for n in polartx.__all__ if n.startswith("bench_")}
registered = {n for n in PRESETS if n.startswith("Bench:")}
assert len(registered) == len(exported)      # ✓ 自维护
```

## 2. 对标测试写"两侧" / Benchmark assertions are two-sided

文献对标预设必须同时断言**下界和上界**：只写 `evm < -30` 时，一个把信号
算成纯噪声的 bug 也能过。要求结果落在发表的量级区间内。

```python
assert -36.0 < res.evm().db < -30.0     # ✓ 落在 Degani'24 的公布类别里
```

## 3. Vendored 代码是"改编副本"，不是 fork / The vendor policy

`src/polartx/vendor/` 下的每个文件都来自 `pll_simulator` 或 `PA_DPD`，
文件头必须带出处：

```python
# Vendored from pll_simulator@d7be4712: src/pllsim/arch/adpll.py
```

`tools/vendor_check.py` 把每个文件分成四类，CI 的 `vendor-drift` job 每次
push 都跑：

- `verbatim` — 与上游逐字一致，直接对拉取到的上游 blob 校验；
- `adapted` — 只有 import 路径等机械改写，同样对上游校验（忽略已声明的重写）；
- `extended` / `extracted` — 有**被批准的**本地改动，在
  `tools/vendor_manifest.json` 里用 `body_sha256` 钉死，并写明 `reason`。

**要改 vendored 文件时**：先问能不能不改（在 `polartx` 侧包一层通常可以）。
确实要改，就更新 manifest 里的 `mode` / `body_sha256` / `reason`，让漂移
检查重新变绿——`reason` 是给未来的人看的，写清楚为什么上游不能满足。

上游 commit 无法访问时（浅克隆、离线），检查器会 **skip**，不会误报 DRIFT。

## 4. 前端不能落后于库 / All surfaces, or none

这个工程有**三个**前端：Streamlit 网页版（`gui/`）、PySide6 桌面版
（`src/polartx/guiqt/`）和 Android WebView 版（`android/`，经
`src/polartx/appbridge.py`）。计算全部住在 `src/polartx/guiutil.py`，可以脱离
三个前端单独测试——**新功能先落在 `guiutil`，再由各前端接线**。

规矩：

1. 新增 `bench_*` 预设必须进 `guiutil.PRESETS`，
   `test_benchmarks.py::test_every_benchmark_is_reachable_from_the_gui`
   会挡住漏接的情况。
2. 新增 GUI 页面要**两个桌面/网页前端都加**，并且两边都有测试
   （`tests/test_gui_web.py` 用 streamlit `AppTest`，
   `tests/test_gui_qt.py` 用 offscreen Qt）。页数不要 hardcode，从
   `PAGES` 推。
3. **Android 的 bridge 方法与页面必须双向对齐**：`appbridge._METHODS` 里多一个
   没人调的方法，或 `app.js` 调一个不存在的方法，`tests/test_android_parity.py`
   都会红（纯文本比对，不需要浏览器或模拟器）。**一个没有调用者的 bridge 方法
   只是半个功能**——它有 bridge 测试，看起来被覆盖了，对用户什么也不做。
4. Android 上刻意缺的功能（RTL 导出、并行蒙卡）记在 `docs/android.md` 与
   `cairn/android-app.md`，**口径差异是要记录的决定，不是留给下一个人发现的缺口**。
5. 报告层画的星座图必须和它打印的 EVM 用**同一个均衡口径**——
   `PolarResult.evm_equalize_default` 就是为此存在的。用不同口径画的图
   会无声地和数字矛盾。

## 5. 数值口径要显式，不要默认 / Make the measurement convention explicit

这个库里出过几次"数字对但口径不对"的事，所以：

- **EVM 均衡**：`scalar`（保留记忆型失真可见）vs `per_tone`（仪器/VSA 口径）。
  哪个都行，但结果对象要通过 `evm_equalize_default` 声明自己是哪个。
- **谱模板**：`polartx.metrics.sem.MaskSpec` 带 `source` 字段。库里自带的
  全部是 `source="stylized"`，`is_conformance` 为 `False`。
  **不要把工程化模板写成认证限值**；真表格请用
  `basis="dBm_in_rbw"` + 指明条款号的 `source` 构造。
- **SEM 是在分辨带宽里积分的功率**，不是一个 FFT bin 的高度。用
  `check_sem`，它的判决与 `nfft` 无关（有测试固化）。

## 6. 引擎有适用域，越界要出声 / Engines declare their validity domain

`ADPLLTwoPoint` 的 `mode="event"` 逐参考周期仿真，只在**每参考周期的相位
推进 ≪ 1 UI** 时有效（BLE 0.8%、EDR 4.2% 可以；LTE 29% 不行）。越界时
它会发 `RuntimeWarning` 并在诊断里给出 `ui_per_ref_cycle_p99`。

新增引擎/近似时照此办理：把适用域写进 docstring，并在运行时检测越界，
而不是让使用者拿到一个安静的错误结果。

同理，指标函数在输入不够时要**抛异常**，不要返回 NaN——
`metrics/dpsk.py` 就是因为这个改过。

## 6.5 声明了的依赖下限，就要真的测 / Test the floor you declare

`pyproject.toml` 写 `numpy>=1.24`，那这句话就是个**契约**。CI 的 `test` 矩阵
永远解析到最新版，所以在 `test-floor` job 出现之前，那个下限**从来没被跑过**。

代价是真的：numpy 2.0 把 `np.trapz` 改名 `np.trapezoid`，一个 vendored 积分器
只用了新名字，于是**每一个窄带 ADPLL 预设**在 numpy 1.x 上都抛
`AttributeError`——37 项测试会红，而 CI 一直是绿的。它最后是在 **Android** 上
暴露的（Chaquopy 给 Python 3.10 的 wheel 是 numpy 1.x，那是第一个真的用下限版本
跑这份代码的地方）。

所以：**改依赖下限就要同时改 `test-floor`**，别让声明和被测的东西分家。

## 7. 提交 / Commits

- 开发分支：`claude/digital-polar-tx-dev-r0c338`。**不要在没被要求的情况下
  推 `main`**——每次都要单独问；**不要主动开 PR**，除非明确要求。
- 提交前：`pytest tests/ -q` 全绿；碰过 examples 就把碰过的跑一遍
  （CI 的 `examples` job 会跑全部 18 个）。
- 提交信息写**为什么**，不写**改了哪些文件**（diff 已经说了后者）。
- 新物理效应/新指标 = 新测试。CI 的 `test` 矩阵覆盖多平台多 Python 版本。
- GitHub 操作走 `mcp__github__*` MCP 工具；这个环境里**没有 `gh` CLI**。

## 8. 代码风格 / Style

跟着周围的代码走：模块级 docstring 说明这个文件在信号链里的位置；
公开 dataclass 的字段用行尾注释注明**单位**（`# [Hz]`、`# [rad]`、
`# fraction of rms`）；向量化优先，逐样本 Python 循环只用在 event 引擎
那种确实需要逐周期状态的地方，并在 docstring 里写明代价。

两条和这个仓库性质直接相关的写作习惯：

- **docstring 或 README 里的物理断言，必须是你这次真的量过的**。这里的数字是
  承重的，不是修辞——写一个没量过的数字，下一个人会拿它去做设计决策。
- **负面结论要写进代码**。已经有好几处 docstring 记着**没成的**东西
  （固定斜率上限下 smoothstep 输给线性插值；不固定 `fs_scale_fixed` 时 ILA
  收益封顶）。保持这个习惯——省掉的是下一个人重做一遍同样的弯路。

### 8.1 静态检查 / The lint gate

`ruff check .`，由 CI 的 `lint` job 背书，版本在 workflow 里**钉死**
（不钉的话新版 ruff 能把没改过的代码判红，和 §6.5 防的是同一类意外）。
规则集与豁免写在 `pyproject.toml` 的 `[tool.ruff]`，选择的理由也在那里：

- **`src/polartx/vendor/` 整个排除**。它是可校验的改编副本（§3），为了消一条
  告警去改上游代码，等于用漂移换整洁；而且每处改动都得在
  `tools/vendor_manifest.json` 里申报，否则漂移检查器直接红。
- **留 pyflakes（`F`），这是真能抓到东西的部分**。未用的导入、赋了值再也没读的
  局部变量，通常是重写时掉下来的引用——和 §1 关心的是同一种失效。接入这条闸门
  时抓到的正是这个：一处 `paths = emit_dpd_rtl(...)` 的返回值从来没被读过
  （文件是靠副作用落盘的），还有两个测试里导入了却压根没用的 preset。
- **E702 / E731 / E741 明确不管**。分号压行的绘图语句、表驱动测试里的一个
  lambda、example 里叫 `l` 的循环变量——纯风格，在这个仓库里零缺陷信号，
  为它们改 example 的叙述节奏不值得。
- **example 里放行 E402**。example 脚本按叙述顺序在用到的那一段就近 import，
  这是 example 的写法，不是错误。

`mypy`（同一个 job，同样钉死版本）跑的是 `src/polartx`，配置同在
`pyproject.toml`。这里的取舍和 ruff 反过来：vendor 树**查类型但不报错**
（`follow_imports = "silent"`）——正是它里面的注解让 `ofdm_ref`、`fir_tx`
在这一侧可检查，而副本内部的告警是上游的事。

> **工具必须装进项目环境，并且用 `python -m mypy` 跑。**
> ```bash
> pip install -e ".[gui,guiqt,dev]"   # dev extra 里是钉死的 ruff + mypy
> python -m mypy && ruff check .
> ```
> mypy 是**从它自己所在的环境**解析 numpy / PySide6 / scipy 的类型的。用
> `uv tool install` 或 pipx 装的独立 mypy 看不见这些包，配合
> `ignore_missing_imports = true` 就把所有涉及它们的东西退化成 `Any`——然后
> 报告 "Success"。接入这条闸门时就踩了：本地 0 错，CI 9 错，**CI 是对的那一边**，
> 因为它把工具和依赖装在同一个环境里。一个检查通过不等于它检查了东西。

第一步只上默认严格度，**没有** `disallow_untyped_defs`：这个库有大量无注解的
函数，一次性要求注解会淹掉真正的信号。想加严的话，加的是
`--check-untyped-defs`（现在 `guiqt/widgets.py:40` 会提醒你它被跳过了）。

写类型的两条本仓库经验：

- **不要用 `object` 当"这里其实是某个具体类"的占位**。`Waveform.ofdm_ref` 和
  `FIRTxPreset.fir_tx` 都曾是 `object` 加一句行尾注释说明真实类型，结果类型
  检查器把它俩之上的**一切**都放过了（`wf.ofdm_ref.tx_symbols` 也一样）。
  写 `if TYPE_CHECKING:` 导入真实类型：运行时零代价，注释变成机器能读的。
- **`Optional` 字段要么窄化要么抛错，不要靠调用方自觉**。缺指标输入时抛异常
  是 §6 的规矩，所以窄化的地方顺手就是那个异常该待的地方
  （`Waveform.require_ofdm_ref()`）。
