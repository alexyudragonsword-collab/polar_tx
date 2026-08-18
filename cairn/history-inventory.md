---
type: project_topic
status: active
summary: "Cairn 初始化前 polartx 已有知识的落点清单（inventory_only）：哪些结论住在哪个文件里，以及哪几条属于'不重做一遍就会重犯'的类型"
tags: [polartx, cairn, 历史清单]
contains: [reference, lesson, open_question]
created: "2026-08-18"
updated: "2026-08-18"
related: []
authoring_mode: ai_generated
---
# 历史知识清单（初始化前）

`migration_mode: inventory_only` 的产物：**只做索引，不改写、不搬运原文**。
Cairn 初始化时项目已有 40 次提交、238 项测试、18 个 examples，历史结论散在若干
文档里；这份清单说明**去哪找**，以及哪些结论值得后续单独成篇。

## 现有文档的落点

| 文件 | 装的是什么 | 对 Cairn 的意义 |
|---|---|---|
| `README.md` | 能力、结果表、建模口径与已知边界、里程碑（均已完成） | 当前真相的**对外**版本 |
| `docs/architecture.md` | 模块地图、三个数据流对象、两个会咬人的实现点、"加东西改哪里" | 改代码前的必读；Cairn 不复制它 |
| `docs/index.html` | 图文设计指南，13 节，中英双语 | 设计取舍的长篇论证 |
| `CONTRIBUTING.md` | 测试/vendor/GUI/口径约定，每条有测试或 CI job 背书 | 规则的完整版；`AGENTS.md` 只摘最易破坏的几条 |
| `CHANGELOG.md` | 按开发轮次的变更记录，**含被推翻的结论** | 最接近 Cairn LOG 的既有物 |
| `ROADMAP.md`（根） | 只写未完成项，分"被外部阻塞/模型缺口/维护/明确不做" | 本项目**不建** `cairn/ROADMAP.md`，就用这份 |
| `examples/ex01`–`ex18` | 可执行的例子，CI 每次 push 全跑 | 结论的可复现证据 |
| docstring | 负面结论的主要栖息地（见下） | 代码即文档，这是本仓的既有习惯 |

## 值得单独成篇的候选（尚未成篇）

这些结论目前只散落在 docstring、测试或 CHANGELOG 里，**具备跨项目复用性**，
是将来接上知识库后的毕业候选。列在这里是为了不让它们被遗忘，不是现在就搬：

1. **EVM 均衡口径 ↔ 实测仪器的对应**。scalar（保留记忆型失真可见）vs per-tone
   （802.11 LTF / 3GPP RS FDE，仪器口径）；两者只在存在频率响应的损伤上分叉。
   这条对任何"仿真数字对不上测试仪"的场合都成立，不限于 polar TX。
2. **能骗人的是 EVM，说实话的是 ACP**。功率域测量没有均衡器，所以频谱再生完全
   暴露且比 EVM 更早报警——在 EDR（ex05）和 WiFi（skew）两个完全不同的场景下
   各自独立验证过一次。
3. **引擎适用域要在运行时检测，不能只写在文档里**。event 引擎的
   ≪1 UI/参考周期条件；违反时告警 + 暴露 `ui_per_ref_cycle_p99`。
4. **ILA/DPD 拟合要求被拟合对象是静态系统**。逐次归一化会让"PA"在迭代间变化，
   收益封顶在 ~4 dB；固定满量程后 EVM 从 −19.8 到 −68.8 dB。
5. **SEM 是分辨带宽里的积分功率，不是 FFT bin 的高度**；且带内相切会把总裕量
   恒钉在 0.00 dB，必须分开报告带外裕量。
6. **vendored 代码的可校验改编副本模式**（出处头 + 分类 + 只对被批准改动钉哈希 +
   CI 漂移检查 + pin 不可达时 skip 而非误报）。这套做法与 polar TX 无关，
   任何要复用姊妹仓代码又不想 fork 的项目都能用。

## 教训类（做错过一次的）

- **硬编码计数会挡住合法改动**：`len(PRESETS) == 14` 和 GUI 页数 `== 3` 各挡过一次。
- **静默 NaN 比抛异常更糟**：会一路传到报告和 GUI 显示成 "nan%"，没有任何线索。
- **"看起来接上了"不等于接上了**：web GUI 曾经只是看起来接上，用 `AppTest` 真跑
  才发现 Arrow 序列化的坑。
- **自己的假设错的概率不低于 vendor 的**：写 vendor 行为测试时三个假设错的是我。
- **假说要证伪就明说**：LTE event 模式的"差一拍"假说被自己推翻，真因至今未定位
  （见 `ROADMAP.md` B1）。

## 开放问题

见根目录 `ROADMAP.md`。这里不复制，避免两份文件各说各话。
