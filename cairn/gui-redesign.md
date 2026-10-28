---
type: project_topic
status: active
summary: "GUI 改版：三端各写一份表单 + 批处理交互换成一份界面规格、一套 HTML/JS 界面三个宿主、实时工作台；决策理由、分阶段计划与每阶段踩到的坑"
tags: [polartx, gui, 前端, android, streamlit, pyside6]
contains: [decision, lesson, pitfall, experience]
created: "2026-10-28"
updated: "2026-10-28（P0 完成）"
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

`test_every_honoured_knob_moves_the_output` 对每个声明的旋钮拨一次、比对输出样本。
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

## 本地环境（pitfall）

容器里 `python` 是 3.11（没装包），`pip` 绑在 `/usr/bin/python3`（3.13）。numpy 2.5.3
的类型存根用了 3.12 的 `type` 语句，而 mypy 目标是 3.10，所以本机 `mypy` 在未改动的
main 上也报存根语法错；CI 的 3.11 装的是 numpy 2.4.x，不受影响。本地用
`mypy --python-version 3.12` 检查自己的代码。
