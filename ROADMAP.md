# 路线图与待办 / Roadmap

**这份文件只写还没做的事。** 已完成的历史在 [`CHANGELOG.md`](CHANGELOG.md)，
当前能力和结果表在 [`README.md`](README.md)。

原始开发计划（Step 0–3 / M2–M6 / T1–T3 / F1–F4 + 五项评审补强）已全部执行完，
所以下面不是"计划的剩余部分"，而是**做完之后才看清的缺口**。

分四类：**被外部阻塞**（需要你提供东西）、**已知模型缺口**（能做，但要先说清代价）、
**工程维护**（低风险、随时可做）、**明确不做**（写下来免得反复讨论）。

---

## A. 被外部阻塞——需要你提供或解除

这几件不是没想清楚，是**这个环境拿不到必要输入**。别让我猜着做。

### A1. 真实的认证限值表格（SEM / ACLR / EVM）★ 优先级最高

内置的所有谱模板都是**工程化简化**，`MaskSpec.source == "stylized"`、
`is_conformance` 为 `False`。这个环境的出口代理封了 3GPP / ETSI / IEEE /
ARIB / 各厂商站点，我拿不到原文，**也不会编造看起来合理的数字**
（`CLAUDE.md` 第 5 条）。

接口已经就绪，缺的只是表格本身：

```python
MaskSpec(points=..., rbw_hz=1e6, basis="dBm_in_rbw",
         source="3GPP TS 36.101 Table 6.6.2.1.1-1")   # is_conformance 自动转 True
```

需要你给的：关心的制式的 SEM 断点表、ACLR 限值、EVM 限值，连同条款号。
给了之后这个库才能做**合规预判**而不只是趋势对比。

### A2. LICENSE 的版权人

现在写的是占位符 `polar_tx contributors`。要换成个人名字还是公司主体，
是你的法律决定，我不替你定。

### A3. 更多实测器件数据

实测通路（`measured.py`）只喂过 OpenDPD 一个器件。多几组不同工艺/频段的
AM-AM/AM-PM 才能说这套建模有普适性，而不是对着一个器件调出来的。

---

## B. 已知模型缺口

### B1. event 引擎覆盖不到 LTE 级带宽

逐参考周期引擎只在每周期相位推进 ≪1 UI 时有效。实测：BLE 0.8%、EDR 4.2%
可用，**LTE-20 是 29%，不可用**——目前的处理是运行时告警 +
`ui_per_ref_cycle_p99`，然后该带宽只能用 `response`（线性化）引擎。

后果是**该带宽下没有杂散/非线性耦合的真值来源**（`CLAUDE.md` 第 7 条：
response 模式不能背书杂散结论）。

我曾假设这是两点通路差一拍造成的，**自己证伪了**（两个方向移位都更糟），
真实原因未定位。要修有两条路：

- 多速率 event 引擎（DCO 侧过采样，参考侧仍逐周期）——工作量大，但这是唯一
  能给 LTE 杂散结论背书的办法；
- 或者接受现状，把"该带宽只有线性化引擎"写成永久边界。

**先定位原因再决定走哪条**，别直接开始写多速率引擎。

### B2. 器件记忆效应是解析式的，不是实测拟合的

`ex09` 给出的差距是明确的：同一组实测数据，**静态极坐标模型 NMSE ≈ −20 dB，
含记忆的 GMP 是 −39 dB**——这 19 dB 就是器件记忆。但链路里的记忆模型
（`chain.memory`、供电推压）是解析构造的，参数不是从实测反演的。

想让链路级预测对得上实测，得把这段补上。

### B3. 远端噪底只有解析预算

−160 dBc/Hz @ 80 MHz 这类指标超出可行记录长度的 Welch 分辨能力，
目前靠 `analysis/responses.py` 的解析噪声预算。**这是物理限制不是偷懒**，
但意味着这类数字没有仿真交叉验证。要验证需要极长记录 + 频段抽取，代价要先算。

### B4. Doherty / 多核合路没有实测对照

`dpa/combiner.py` 的效率是从负载调制物理**导出**的（不是拟合的），这一点比
拟合模型强；但整个模型没有和任何实测 Doherty DPA 比对过。

### B5. Android 版真机验证只做了一轮

第一次侧载就抓到一个 CI 完全看不见的 bug（运行按钮全无反应，见
`cairn/android-app.md`）。修完加了 jsdom 真 DOM harness，但**修完之后还没有
再侧载确认过**——尤其是编译版的 `import` 在真机上是否成功，至今没有直接证据。

还没量过的：手势手感、刘海与安全区；双抽头 FIR（osr=50）在 x86 runner 上
2.1 s，手机上是否需要更小的默认值。

### B6. 剩 24 处 docstring

都是 GUI 内部件和父函数已解释的嵌套闭包。低价值，列出来只是为了闭环。

---

## C. 工程维护

### C1. 版本与发布流程

**打包元数据已经补齐**（2026-09-12）：classifiers、`[project.urls]`、keywords，
以及 PEP 561 的 `py.typed`（实测在 wheel 里）；版本号在 `pyproject.toml` 与
`polartx.__version__` 两处的一致性由 `tests/test_packaging.py` 卡死。

**剩下的才是这一条的本体**：`pyproject.toml` 仍停在 `0.1.0`，没有 git tag、
没有 release、没上 PyPI。CI 的 `build` job 已经在产 sdist + wheel，缺的是
发布这一步和版本策略——姊妹库 `pll_simulator` 用的是"release notes 触发
auto-release"，可以照搬，但要先决定这个库要不要给外部用。

另：`LICENSE` 的版权人还是 "polar_tx contributors"（见 A2），对外发布前要定。

### C2. Windows exe 与 Android APK 未接入发布流程

两条 workflow 都在真 CI 上成功跑过（`windows-exe` ×2、`windows-exe-nuitka` ×3，
均为手动 `workflow_dispatch`），产物正常。但它们和 release 没有联动——
打 tag 不会自动出 exe，也不会出 APK（`android.yml` 同样是手动触发）。

### C3. ~~`np.trapezoid` 的修法属于上游~~ → 已了结（2026-09-13）

原条目写的是"同一个问题上游 `pll_simulator` 也有，应当推回上游、这里恢复
verbatim"。**去查证时发现前提已不成立**：上游在 `1b0f308`（"a real numpy
floor"）里自己修了，写法是 `vars(np).get("trapezoid") or vars(np)["trapz"]`，
两侧都走 `vars(np)`，比本仓当时的 `getattr(...) or np.trapz` 更稳——后者的
`np.trapz` 是静态属性读取，numpy 2 里该属性不存在，类型检查器会判错（本仓
`selector.py` 就撞到过）。上游还配了跑 numpy 1.24.4 的下限 job，并校验装到的
确实是下限版本。扫过上游 `src/` 其余位置，没有同类残留。

所以没有东西可推回。后半句"恢复 verbatim"已随下面的子树推进完成：`jitter.py`
不再有本地改动，manifest 条目从 3 条降到 2 条。

### C5. vendored `adpll.py` / `frac.py` 仍钉在 `d7be4712`

2026-09-13 把 `vendor/pllsim/` 的其余 39 个文件推进到上游 `931cfaf`（八个月的
演进，+2684/−516，19 个文件有实质变化；另新增上游依赖的 `core/jit.py` 与
`core/boundaries.py`）。**16 个 preset + 3 个带损伤配置的报告逐值比对完全一致**，
套件 272 passed 不变。

这两个文件没有跟上，原因是结构性的、不是没排上：

- 上游把逐周期循环搬进了 `@kernel` 装饰的函数（`core/jit.py`：装了 numba 就编译，
  没装就纯 Python 跑，两条路径要求**逐位一致**，并明令 per-cycle 路径里不许有
  numpy 数组操作）。
- 本仓在 `adpll.py` 上的扩展恰恰是每参考周期回调一个 Python 对象
  （`dp_cal.step(e_ui, mod_freq[n])`，在线两点增益校准），**装不进这种 kernel**。
  要跟进就得把 LMS 状态摊成浮点数组穿过 kernel 签名重写一遍——那是改物理路径，
  得单独做、单独验证。
- `frac.py` 是从上游 `cppll.py` 抽出的 37 行，新版 `cppll.py` 也大改了，要重抽。

因此 pin 是**混合**的：39 个文件 @`931cfaf`，2 个 @`d7be4712`。CI 的 sibling
checkout 因此必须 `fetch-depth: 0`（浅 checkout 解析不到另一个 commit，
`vendor_check` 会把这两个文件**跳过**——正好跳过唯一两个有本地改动的文件），
并且现在传 `--fail-on-skip`：跳过即失败，不再当作通过。

### C6. Android 编译集需要重测

`docs/android.md` 的 **42 / 101 个模块编译** 是 2026-08-25 那次真机构建的实测值。
vendored 子树推进后包里是 103 个模块，**分子没有重测**——编译集是由"删掉 `.py`
后套件仍过"证明出来的，不是算出来的。下次手动跑 *Android APK* workflow 时顺带
重新记一次。历史记录（CHANGELOG、`cairn/`）里的 42/101 是当时的实测，不要改。

### C4. `mode="event"` 的性能

Python 逐周期循环 ~1–2 Mcycles/s。这是**故意的**（`CLAUDE.md`：真正需要逐周期
状态的地方才用 Python 循环），但如果 event 模式的使用频率上来了，
向量化/numba 化值得重新评估。

---

## D. 明确不做

写下来是为了**不要每隔几轮就重新讨论一次**。要改主意可以，但请明确说。

- **信道编码 / 接收机链路**。这是 TX 损伤库；没有任何指标在比特映射之后测量。
  加了编码，EVM 和 BER 的因果关系就糊了。
- **替代认证测试仪**。就算 A1 给了真表格，这个库也只做预判，不出合规报告。
- **把 vendor 变成 fork**。`src/polartx/vendor/` 是可校验的改编副本，
  漂移检查是它的价值所在。要长期分叉，正确做法是把改动推回上游。
- **GUI 里再加架构**。架构开关是 `PhaseModulator`（`docs/architecture.md` §1）；
  新架构走那个接口，GUI 自动继承。

---

## 怎么用这份文件

改动落地后，把对应条目从这里删掉、在 `CHANGELOG.md` 里记一笔。
两边同时留着同一件事，是这份文件开始烂掉的第一个征兆。
