---
type: project_topic
status: active
summary: "polartx 的静态检查闸门（ruff + mypy）为什么这样划范围，以及行为不变重构的验证协议：dump-比对，而不是'套件还绿'"
tags: [polartx, 工程纪律, ruff, mypy, 重构]
contains: [decision, lesson, experience, open_question]
created: "2026-09-12"
updated: "2026-09-12（当日修订：见「本地 mypy 跑在空气上」）"
related: []
authoring_mode: ai_generated
---
# 静态检查闸门与重构纪律

规则全文在 `CONTRIBUTING.md` §8.1（那是工程资产）。这里只放**过程知识**：
范围是怎么划的、划错会怎样、以及重构时凭什么敢说"行为没变"。

## 形成背景

2026-09 的一次工程体检发现：仓库有 277 项测试、7 个 CI job、vendor 漂移检查器，
却**一个静态检查都没有**——`pyproject.toml` 只有 pytest 一节。不是因为代码脏
（默认规则集只报 39 处，其中 11 处是纯风格），而是因为没人加。结论是加闸门的
收益不在"当下清理"，而在"没有闸门就不会红"。

## 决策

- **vendor 树在两个工具里待遇相反**：ruff 整个排除，mypy `follow_imports = "silent"`
  （查类型、不报错）。理由不是偷懒：`src/polartx/vendor/` 是可校验改编副本，
  任何只为消告警的改动都得在 `tools/vendor_manifest.json` 里申报，等于用漂移换
  整洁；但它的**注解**正是本侧能被检查的前提（见下面 `object` 那条），所以类型要
  读、错误不看。一共 2 处 ruff、17 处 mypy 错落在里面。
- **只留有缺陷信号的规则**。pyflakes（`F`）留全；`E7` 留着是为了 E711/E712/E713/
  E714/E722 这几条真的；E702（分号压行的绘图语句）、E731（表驱动测试里的一个
  lambda）、E741（example 里的 `l`）明确不管——为它们改 example 的叙述节奏零收益。
- **CI 里钉死工具版本**（ruff 0.15.8 / mypy 1.19.1）。不钉的话新版本能把没改过的
  代码判红，和 §6.5 的依赖下限 job 防的是同一类意外。
- **mypy 第一步不上 `disallow_untyped_defs`**。本库无注解函数很多，一次性要求会
  淹掉真信号。下一步该加的是 `--check-untyped-defs`。
- **lint job 装 GUI extras**。不装的话 PySide6/streamlit 退化成 `Any`，CI 检查得比
  开发机还少——闸门弱于本地是错的方向。

## 经验与教训

- **本地 mypy 跑在空气上，是 CI 抓出来的。**（2026-09-12 当日修订：本文最初写着
  "mypy clean over 49 files"，那个结论是假的。）本地的 mypy 是 `uv tool install`
  装的，有自己独立的 Python 环境，**看不见项目的 numpy / PySide6 / scipy**；配合
  `ignore_missing_imports = true`，所有涉及这些库的东西退化成 `Any`，于是报
  "Success: no issues found"。CI 把 mypy 和依赖装在同一个环境里，报 **9 个错**——
  两边的 numpy 都是 2.4.6、mypy 都是 1.19.1，差别只在"工具能不能看见依赖"。
  这正是本仓库历史上反复出现的那类失效：**一个检查通过，不等于它检查了东西**。
  堵法是结构性的：工具版本收进 `pyproject.toml` 的 `dev` extra，
  `pip install -e ".[gui,guiqt,dev]"` 之后用 `python -m mypy` 跑，和 CI 同环境。
  那 9 个错里 8 个是真该修的（numpy 2 的 `np.trapz` 静态属性、`list` 被重新绑定成
  ndarray、异构 meta dict 被推断成 `dict[str, float]`、`PATTERNS` 的 None 哨兵、
  `phase_ideal` 的 Optional），1 个是 PySide6 stub 没声明转发枚举
  （`QTableWidget.NoEditTriggers` 运行时在、stub 里没有，改用
  `QAbstractItemView.EditTrigger.NoEditTriggers`，并实测建表仍正常）。
- **不要用 `object` 当"这里其实是某个具体类"的占位。** `Waveform.ofdm_ref` 和
  `FIRTxPreset.fir_tx` 都写着 `object` 加一句注释说明真实类型，结果 9 个非 vendor
  mypy 错里 6 个出自这一条：`object` 让**穿过该字段的一切**也不可检查，
  `wf.ofdm_ref.tx_symbols` 在三个模块里都是裸的。`if TYPE_CHECKING:` 导入真实类型：
  运行时零代价，注释变成机器能读的东西。实测佐证——改完 mypy 对该字段的称呼从
  `object` 变成 `OFDMWaveform | None`。
- **窄化 Optional 的地方，正好是异常该待的地方。** 顺着类型检查器指的三处，
  发现 `cfg.seed + 777`：vendored OFDMConfig 允许 `seed=None`（表示"取新熵"），
  于是**带 pilot 或 preamble 的波形在 seed=None 下会死在这行算术里**，而普通波形
  没事。同理 `p.tx.phasemod.cfg` 穿过抽象接口取具体配置——`ADPLLTwoPoint` 根本
  没有 `.cfg`（它在 `.pll.cfg`），把窄带 preset 喂给蒙卡会在抽样中途
  AttributeError。两处都改成"明确窄化 + 抛带话的异常"。
- **prose 里的计数是会假的，而且两个方向都会假。** README 写"230+ 项测试"：按
  pytest 实收是 277（过期），按测试函数数是 229（**声明本身不成立**）。现在由
  `tests/test_docs_consistency.py` 卡住，并且**按 AST 数函数、不数 pytest 实收**——
  实收数取决于装了哪些可选依赖（没 PySide6 时 `test_gui_qt.py` 一个都不贡献），
  那是机器的属性，不是仓库的属性。
- **文档一致性测试要卡两个方向**：模块没进地图是遗漏，地图写了不存在的文件是
  改名没跟上。只补 14 行而不加测试，只是把时钟拨回去——`docs/architecture.md`
  已经漂过一次了。

## 行为不变重构的验证协议

拆 `guiutil.run_chain_report`（168 行、圈复杂度 D(29) → A(3)）时用的办法，
以后同类重构照用：

1. **拆之前**先把全部 16 个 preset ＋ 3 个带损伤的配置跑一遍，dump 成 JSON：
   metrics 全量键值、图上每个 axes 的 title/label/xlim/ylim、每个 artist 的数据
   哈希（脚本不进仓库，属一次性工具）。
2. 拆完再跑一遍，**逐值比对必须完全一致**（这次 19 个配置全等）。
3. 然后才是三个前端的测试（78 项）＋ 全量套件 ＋ 跑到报告层的 example。

**"套件还绿"不足以背书这类重构**：套件里没有任何测试会因为图换了坐标轴范围、
或星座图用了另一种均衡口径而失败。这次拆的收益恰恰在这条线上——计算与绘图分开
之后，图上画的点和旁边印的 EVM 只可能来自同一次均衡（`_Constellation` 由指标阶段
算出、图阶段只负责画）。

## 待办

- vendor 陈旧引脚 23 个、累计落后上游 1193 行，校验器目前只当 advisory
  （`--strict` 才红）。方向是把本侧的修法推回上游（ROADMAP C3 的 `np.trapezoid`
  就是一例），不是追平副本。
- `guiqt/` 是覆盖率最低的非 vendor 目录（widgets 65%、pages 76.5%），Qt 侧只有一个
  smoke 测试。
