---
type: project_topic
status: active
summary: "GUI 改版：三端各写一份表单 + 批处理交互换成一份界面规格、一套 HTML/JS 界面三个宿主、实时工作台；决策理由、分阶段计划与每阶段踩到的坑"
tags: [polartx, gui, 前端, android, streamlit, pyside6]
contains: [decision, lesson, pitfall, experience]
created: "2026-10-28"
updated: "2026-10-29（P4a 其余页面迁入）"
related: [android-app.md]
authoring_mode: ai_generated
---
# GUI 改版

提案（含现状截图、真实数据驱动的原型）：https://claude.ai/artifact/BeVCjVUvLK2Se2vXtPDjAt
（私有页面）。这里只记过程知识。

## 决策（2026-10-28，用户拍板"做吧"，按提案的推荐）

- **一套界面，三个宿主**。Android 的 HTML/JS 页面长成唯一的界面；手机宿主不变
  （Chaquopy + WebView），桌面与浏览器由 `polartx-gui` 起标准库本地服务、用系统浏览器
  应用窗口打开。Streamlit 与 Qt 页面在 P4 退役。理由：三份手写前端已经漂移且算错，
  "前端同步"是这个仓库最贵的规矩，合成一份文件后它变成结构性保证；代价是与 pllsim 的
  三前端结构不再一致、CI 的 GUI 测试与 Windows 打包要改。
- **桌面先不要原生窗口**：Qt WebEngine 会让 exe 大很多、打包更麻烦。
- **分阶段**：P0 六处修正 → P1 界面规格层 → P2 新工作台 + 本地宿主 → P3 基线/扫描/
  历史/拓扑对比 → P4 迁移其余页面、旧前端退役。每阶段单独 PR，CI 全绿后合入。

## P0 的两处算错是怎么来的（pitfall）

三端各自按预设名子串（`"WiFi" in name`）决定发哪些覆盖参数。名字是给人看的：
"WiFi 160 MHz (RF-DAC)" 也含 "WiFi"，于是 DTC 位数旋钮（11）被当作它的 DAC 位数发过去，
而它自己的默认是 12；"WiFi 160 MHz (outphasing)" 拿到了 skew，而 outphasing 没有包络
通路、库里静默中和。现在由 `guiutil._HONOURS` 一处声明，`chain_overrides` 过滤，三端
只调它。RF-DAC 在那个工作点上 11 bit 与 12 bit 的 EVM 都是 −40.0 dB（噪声限），所以
没有任何数字露馅——这类错只能靠"声明 + 拨动测试"抓，读输出是读不出来的。

## 经验："拨得动"要按物理来测

`test_every_honoured_knob_moves_the_output` 对每个声明的旋钮拨一次、比对输出样本
（P1 起由 `test_guispec.py::test_every_knob_moves_the_output` 接替，覆盖全部旋钮规格）。
第一次跑 BLE / EDR 的 `loop_bw` 红了：噪声关掉时 200 kHz 与 100 kHz 的输出差 6e-14。
这不是空旋钮——两点调制理想匹配时，环路带宽只整形噪声与失配（开噪声后差 0.05，
5 % 增益误差下 EVM 5.8 % → 3.4 %）。所以 `loop_bw` 在开噪声、同 seed 下比较；测试里
把这条物理写成 `_SHAPES_ONLY_NOISE`，而不是放宽容差。

BLE GFSK 上的 skew 则是真的空旋钮（恒包络，0→10 ns 逐位相同），从 BLE 的旋钮表里去掉；
EDR 是准恒包络（0.2 相对差），保留。

## 经验：Bluetooth 的邻道要按带内发射口径

通用 ACLR 在一个波形带宽处取邻道。GFSK 自身频谱还在那里，BLE LE-1M 读 −22.2 dBc；
同一次运行在 ±2 / ±3 MHz 是 −56.0 / −60.5 dBc。现在 Bluetooth（GFSK，或 1 Msym/s 的
DPSK）报带内发射，LE 2M 的偏移按 `ble_mask` 的同一缩放放到 ±4 / ±6 MHz。EDGE 也是
DPSK 但 270.833 ksym/s，不是 Bluetooth，保留 ACLR。频偏均值与漂移是本 burst 自身载荷上
的，不是认证码型（11110000 / 10101010）定义的 Δf1avg / Δf2max，指标名不借用认证名字。

画频偏轨迹时注意：默认载荷是 PRBS，LFSR 刚起步的前 ~100 bit 几乎全 0，从 burst
开头取窗只看到一个单音。窗口取在 burst 中段。

## P1：界面规格层（`guispec.py`）

**决策**：前端契约放在独立的 `guispec.py`（只描述：旋钮、指标含义、图数据格式、
matplotlib 渲染），计算仍在 `guiutil`。旋钮的默认值**不手填**，从预设函数签名读出再换成
显示单位（`nr_dtc` 的 dither 经 `**kw` 落到 `wifi_dtc`，所以查找链是有序的函数元组）。
两条测试把规格钉在库上：表单全取默认时输出与不加覆盖的预设逐位相同；每个旋钮拨一次
输出样本必须变。break-check：把 RF-DAC 的默认查找顺序换成先 `wifi_dtc`（读到 11）→ 红；
把 skew 加回 outphasing → "ignored" 告警 → 红。

**经验："只在开噪声时起作用"是一类，不是例外**。`loop_bw` 之后第二个是 DTC 抖动：
它本身就是噪声源，噪声关掉就不存在。标志从 `shapes_only_noise` 改名为 `noise_only`，
覆盖"噪声源本身"和"两点匹配时只整形噪声的环路"两种物理。

**坑：报告层吞掉了引擎越界告警**。LTE-20 用 event 引擎会告警"越出适用域，结果会悲观得多
（−15 dB vs response 的 −37 dB）"，但 `run_chain_report` 之上没有任何前端看得到告警。现在
报告带 `"warnings"`，三端都显示；同时带 `"conventions"`（均衡口径、mask 来源、ADPLL 引擎），
口径从此不只在星座图标题里。

**图数据**：报告返回 `"plots"`，PNG 由它画出（测试逐条比对 matplotlib 线条与数据），手机
将来自己画。频谱按 max-hold 抽到 ≤2048 点（判决在全量数据上先算好，只稀疏画法）；
过桥时再四舍五入到 5 位有效数字：WiFi 160 一次运行不含 PNG 约 130 KB。`figure=False`
跳过 matplotlib，给实时重算用。

## P2：一个页面，三个宿主

**决策：页面进 Python 包**。原来页面只在 `android/app/src/main/assets/www`，桌面拿不到。
现在放在 `src/polartx/webui/www/`（package-data，wheel 与 sdist 都带）：手机由 Gradle
`assets.setSrcDirs(listOf("../../src/polartx/webui"))` 打进 APK，桌面与浏览器由
`polartx-web`（`webhost.py`，标准库 `ThreadingHTTPServer`，只绑 127.0.0.1，调用串行化——
有几个报告仍用 pyplot 出 PNG，pyplot 不是线程安全的）提供。页面里没有注入的 `window.host`
时自动走 `fetch("call")`，所以 `app.js` 不分宿主。

**工作台**：控件全部由 `list_presets().specs` 生成（标签、单位、范围、默认值都来自 Python）；
改参数 250 ms 防抖后重算，同时只有一个请求在途，回来时设置又变了就再跑一次（不排积压）；
`png=False`，图由 `charts.js` 在 canvas 上画报告给的数据，读数取曲线上真实的采样点。
其余四个标签（FIR、合路、选型、良率）暂留原表单与 PNG，P4 迁移。

**验证**：`tests/test_webui.py` 起本地宿主、用真 Chromium 驱动真 bridge——skew 0 → 0.3 ns
时 EVM 恶化 >10 dB 且 mask 由通过变失败；RF-DAC 只给 DAC 位数（12）不给 skew；LTE event
引擎的越界告警出现在页面上；语言切换不丢控件；390 px 宽无横向滚动、抽屉导航与旧标签可用。
CI 的 test-gui 装 Chromium 并设 `POLARTX_REQUIRE_BROWSER=1`，跳过即失败。jsdom harness
改为按 `index.html` 的 `<script>` 顺序执行全部脚本（只跑 `app.js` 时 `Charts` 未定义）。
break-check：生成的控件不触发重算 → harness 红；控件不用规格默认值 → 浏览器测试红。

**经验：编译版 APK 读得到默认值**。`guispec` 靠 `inspect.signature` 读预设默认值，而编译版
把 `presets` Cython 化。按 `android_wheel.py` 的同样参数（`-3 --no-docstrings`）本机编译
`presets`、删掉 .py 后对比：62 个旋钮默认值与纯 Python 版完全一致（Cython 3 默认 binding）。

## P3：基线、扫描、历史、拓扑对比

**决策：计算只是编排**。`guiutil.run_sweep`（一个控件扫一遍，其余保持当前值）和
`run_compare`（几个预设同 seed、同一组共享控件）每个点都是普通的 `run_chain_report`，
测试要求扫描与对比结果和手工逐点单跑**逐项相等**。基线与历史纯在页面里：钉住的是一次
运行的报告，指标卡按 `metric_info.better` 给 Δ 着色，图在同 id 的面板下叠画基线；历史
最多 8 条，存 localStorage（读写都 try/catch，存不了就只在内存里）。

**经验：扫描结果里指标集会变**。口径残差（"按子载波均衡后的 EVM"）只在差距 >1 dB 时才
报告：skew 0 时没有、0.1 ns 起才有。所以扫描按并集取键、缺的点给 `None`，图上断线，而不是
把第一点的键集当全部。

**坑：两个表单复用一个控件生成函数会造出重复 id**。对比页的 skew 控件与工作台的同为
`#knob-skew_ns`，`getElementById` 只认第一个，浏览器测试里 RF-DAC 那条（"不该有 skew 控件"）
因此红了——不是 RF-DAC 的问题，是对比页的。现在控件 id 带表单前缀，浏览器测试顺带断言
全页 id 唯一。

**迁移期规矩**：新交互（基线、扫描、历史、对比）只做在新页面里；Streamlit 与 Qt 在 P4
退役，不再为它们补这些功能。

## P4a：其余页面迁入，旧标签改画图数据

**页面分组**：设计（工作台、拓扑对比、选型）、验证（校准、实测数据、合路与陷波）、量产与
交付（良率、RTL 导出）。FIR 陷波并入"合路与陷波"。旧的 FIR / 合路 / 选型 / 良率报告也改为
返回图数据（PNG 由它画出，测试逐线比对），页面在 canvas 上画；`charts.js` 为此加了对数横轴、
反向横轴（回退轴从右往左读）、分类刻度和柱状图。

**效率进工作台**：调制平均效率与平均回退成为每次运行的指标（ex10 的结论落到最常看的地方），
CFR 的效率–EVM 取舍直接用"扫描 CFR、看效率"得到，不需要单独的页。BLE 恒包络 85 %、无回退；
WiFi 160 约 43 %、回退 8.9 dB；RF-DAC 约 28 %。

**选型 → 工作台**：选型报告带 `workbench_preset`（最近的注册预设），页面给"在工作台打开"。

**桌面专属方法**：RTL 导出只在电脑上有意义，由 `webhost.DESKTOP_METHODS` 提供，页面在没有
注入 `window.host` 时才显示该标签；parity 测试把桌面方法纳入双向检查，jsdom harness 断言
手机宿主下该标签隐藏。

**坑：搬卡片时把 FIR 表单嵌进了 `cb-out`**。拼接 HTML 时找错了闭合标签，FIR 的整张表单落进
合路结果的输出容器里——第一次跑合路就把 FIR 控件冲掉。静态检查全过（id 都在），是 jsdom harness
的"跑完后控件还在吗"那一条抓到的；修完后加了一次全页标签平衡检查。

## 本地环境（pitfall）

容器里 `python` 是 3.11（没装包），`pip` 绑在 `/usr/bin/python3`（3.13）。numpy 2.5.3
的类型存根用了 3.12 的 `type` 语句，而 mypy 目标是 3.10，所以本机 `mypy` 在未改动的
main 上也报存根语法错；CI 的 3.11 装的是 numpy 2.4.x，不受影响。本地用
`mypy --python-version 3.12` 检查自己的代码。

别给 pytest 加 `-q`：pyproject 已有 `addopts = "-q"`，叠成 `-qq` 就不打印计数行了
（P1 里跑过一次"全绿却没数字"，pllsim 的 AGENTS 早有这条）。

仓库根有个被 `.gitignore` 忽略的 `build/` 目录：在仓库根跑 `python -m build` 会把它当成
同名包导入而失败（"No module named build.__main__"）。从别的目录跑并把仓库路径作参数。
