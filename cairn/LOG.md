# Project Cairn 日志

本文件按倒序记录实质性进展——最新的一条在这行下面。每条保持简短，只写摘要和指针；结论沉淀进 `cairn/<topic>.md`。

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
