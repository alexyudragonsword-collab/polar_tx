# Project Cairn 日志

本文件按倒序记录实质性进展——最新的一条在这行下面。每条保持简短，只写摘要和指针；结论沉淀进 `cairn/<topic>.md`。

## 2026-08-18 · Project Cairn 初始化

- 在已成熟的项目上做 retrofit：`polartx` 此时已有 238 项测试全绿、18 个 examples、两个 GUI、40 次提交。
- 配置：`git_policy: track`（本仓文档全公开，知识层同样公开最一致）、provider **暂缓对接**、`language: zh`、`migration_mode: inventory_only`。
- 原有的 `CLAUDE.md` 不是丢弃而是**升格**：内容整体搬进 `AGENTS.md` 并补上 Cairn 的规则层，`CLAUDE.md` 退化成一行 `@AGENTS.md` 存根。差异见本条下面的"初始化对 CLAUDE.md 的影响"。
- 本项目**不建** `cairn/ROADMAP.md`：根目录已有 `ROADMAP.md`（只写未完成项），再建一份会重演"两个文件抢同一个名字"的老问题。AGENTS.md 的阅读顺序里直接指向根目录那份。
- 历史清单（`inventory_only` 的产物）：见 `cairn/history-inventory.md`。
- 细节见 `AGENTS.md` 与 `.cairn/config.yaml`。
