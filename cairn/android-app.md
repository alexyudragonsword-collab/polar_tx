---
type: project_topic
status: active
summary: "polartx 的 Android 面（Chaquopy + WebView）：可行性闸门为什么落在 Python 3.10、编译版实测买到与没买到什么、以及手机与桌面之间刻意的口径差异"
tags: [polartx, android, chaquopy, cython, 前端]
contains: [decision, experience, lesson, open_question]
created: "2026-08-25"
updated: "2026-08-25"
related: []
authoring_mode: ai_generated
---
# Android 面

面向使用者与构建者的完整说明在 `docs/android.md`。这里只放**过程知识**：
当时为什么这么选，以及哪些是决定、哪些还是缺口。

## 决策记录

### 可行性闸门靠的是姊妹库的实测，不是自己推断

Chaquopy 的 Python 版本由最重的二进制依赖决定。本仓的出口代理封了
`chaquo.com`（403 policy denial），拿不到它的包索引，**也不去猜**。

改用一条更硬的证据：姊妹库 `pll_simulator` 用**完全相同的三个二进制依赖**
（numpy/scipy/matplotlib，传递性带上 contourpy/fonttools/kiwisolver/pillow）
在 Chaquopy Python 3.10 上构建，*Android APK* workflow 跑过 14 次，最近一次
两个 ABI 交叉编译、ELF 校验、两个 Gradle 构建全部通过。

这也意味着**本地跑不了 Gradle 构建**，那是 CI 的事。

### 编译集是测出来的，不是选出来的

`--compile` 故意没有"全部"默认值：选定的集合等于一个断言——删掉这些 `.py`
之后套件仍然通过。所以按这个顺序做：建 wheel → 干净 venv 安装 → 确认磁盘上
**没有任何 `.py` 可回退** → 跑全量。

结果：**42 / 101 个模块**，**242 passed / 12 skipped**。

过程中测掉一个真风险：`presets` 的 `make_waveform` 靠 `inspect.signature`
分派 burst 长度关键字，Cython 化后签名内省**仍然可用**。这条是测出来的；
如果不测就把 `presets` 排除掉，会平白少保护一块，理由还是假的。

### 三个模块刻意不编译，各有理由

- `vendor/` —— 它的价值就是**可审计的改编副本**（出处头 + 漂移检查）；
  而且那不是 polartx 自己的 IP。
- `guiqt/` —— 桌面专用，手机上不导入。
- `appbridge.py` —— 它的**整个方法表在 `app.js` 里是明文**，编译它保护不了
  任何尚未公开的东西。这里用的是技能文档那条判据：不要为**保护的表象**付出
  构建面积。

## 经验

### 没有对照组的阴性结果什么都不说明

验证"编译到底藏住了什么"时，第一次跑对照组就**失败**了——我拿去搜未编译
模块的短语大小写写错，于是"编译模块里搜不到散文"这个阴性结果当时毫无意义。
改对之后才成立：

| 检查 | 结果 |
|---|---|
| 对照：未编译 `presets.py` 的散文 | **仍在** |
| 已编译 `chain.so` 的 docstring/注释 | **全无**（5 个短语全落空） |
| 已编译 `chain.so` 的字面常量 | **全在**（`2e6`/`50e-9`/`0.15` 各精确命中 1 次） |

所以对外只能说：**把读算法的成本从"解压即读"抬到"反汇编"**。不是许可证
校验，不是数据保护，也不掩盖 UI 显示的任何东西。

### 写完门禁要把它检查的东西弄坏一次

`inspect_apk.py` 是唯一挡住"两个 APK 其实是同一个"的东西（两次构建共用工作区，
真实失败模式是第二次复用第一次的 pip 输出，构建日志对此只字不提）。
用合成 APK 验了五个方向：该绿的两个绿；"解释版冒充编译版"、"编译版冒充
解释版"、"源码盖住 `.so`"三个都红。

### 测试自己也会撞上自己的散文

`test_the_app_declares_no_permissions` 第一版直接搜字符串 `uses-permission`，
结果被 manifest 里**解释这条政策的注释**触发。改成先剥注释再找真标签。
这类"检查撞上自己的文档"很容易反过来被当成被检查对象的问题。

### 交叉编译才会暴露的坑：`cpow`，以及"排除坏模块"是个假解

第一次真 CI 构建挂在 `export/rtl.c`：Cython 把 `x ** y` 降级成 `cpow()` 却不
include `<complex.h>`，NDK 的 clang 把隐式声明当**错误**。`--host` 试跑看不到，
因为 glibc 的头链间接带进了声明。

第一反应是"把 `export` 移出编译集"——它本来就是桌面专用，手机根本不导入，
和已排除的 `guiqt/` 同一条理由，听起来还顺理成章。**但那是错的**：扫过全部
42 个模块的生成 C，**13 个**真的调用 `cpow`，clang 只是死在第一个上。移掉一个
只会让下一个接着倒。

教训：一个看起来"有原则"的修法，如果没先量清楚问题范围，会把一次失败变成一串
失败。查一遍生成的 C 只花了一分钟。

`-Wl,--no-undefined` 只加在交叉路径上，也是撞出来的：主机构建**故意不链接
libpython**，在那里要求全部符号解析会直接炸在 Python C-API 上。

### 第二次也修错了：显而易见的头文件修法解决不了"函数根本不存在"

第一轮我加的是 `-include complex.h`，并在文档里写成"对照验证过、加上就干净"。
**第二次 CI 用同一条错误否掉了它**：命令行带着那个 flag，clang 仍说请 include
`<complex.h>`。

真正的原因是**Bionic 在这个 API 级别根本没有 `cpow` 这个函数**——头是包进来了，
里面没有。我当时"验证"的是那个 flag 在 **glibc** 上让声明出现，而缺的从来不是
glibc 上的声明。**对照做在了错误的平台上，所以那次验证是空的。**

真开关是 `-DCYTHON_CCOMPLEX=0`：Cython 改用自带的复数实现，只依赖实数 libm，
与 API 级别无关。这次按**符号**验证（不加引用 `cpow`、加了 42 个 `.so` 一个复数
符号都没有），并确认受影响函数两种实现输出逐位相同。

教训比上一条更具体：**交叉编译的问题，验证必须做在目标那一侧的约束下**。
在主机上"证明"一个 flag 有效，对目标平台缺失的符号毫无预测力——要么找到能在
主机上复现目标约束的做法（这次是查目标文件的未定义符号），要么就承认它未经验证。

### 双语页面 + `read_text()` = 只在 Windows 上红

推 Android 那一版把 CI 弄红了，而我当时只报了本地 264 passed **没去看 CI**——
这是流程上的漏，不是技术问题。

技术上：`test_android_parity.py` 用裸 `Path.read_text()` 读页面文件，它走的是
**locale 编码**，Windows runner 上是 cp1252，而这些文件装着 app 的中英双语串。
10 个 job 里 9 个绿，只有 `test (windows-latest, 3.11)` 红，14 项全挂在
`UnicodeDecodeError`——报的错和被测的东西毫无关系。

本地用 `encoding="cp1252"` 复现到了 CI 报的同两个字节（`0x8f`/`0x90`）。
顺带发现更阴的一点：`style.css` 在 cp1252 下**不报错，直接乱码**——
错误至少还会响。

改成模块级 `read()` helper 统一走 UTF-8，以后再加读取点也不会漏。

### 静态检查看不见"运行时被销毁"——第一次上真机就栽在这

构建全绿、两个 APK 都装上了，**点运行按钮毫无反应**。没有遮罩、没有错误卡、
什么都没有。

原因：`applyLang()` 用 `el.textContent = …`，而 **textContent 会替换全部子节点**。
页面里有 21 个 `<label data-zh=…>` 直接包着自己的 `<input>`/`<select>`，
于是 boot 结尾那次语言刷新把**所有控件从 DOM 里删掉了**；点击处理器随即在
`$("ch-preset").value` 上对 null 取属性、同步抛异常——按钮因此完全静默。

**我的整套 parity 测试全都通过了**，因为它们查的是"id 在 index.html 里存在"。
id 确实在，只是运行零点几秒后就不存在了。这类 bug 纯文本检查**结构上看不到**。

补的是两层，而且都按规矩先弄坏一次确认会红（回退单个 label，两层同时变红）：

1. **静态不变量**：带 `data-zh` 的元素不得包含任何元素（用 HTML 解析器判，
   不是正则——我第一版正则报了 37 个误报）。标签文字放进内层 `<span>`。
2. **真 DOM harness**（`tests/android_page_harness.js`，jsdom）：执行 `app.js`、
   跑 boot、逐个点击运行按钮，断言调用真的到达 bridge 且参数可用；它当场
   复现了真机现象，包括那条 `TypeError: … reading 'value'`。

CI 里 harness 是**独立一步直接调 node**，不走会 skip 的 pytest 包装——
skip 掉的绿和真跑过的绿长得一模一样。

教训：`docs/android.md` 里"CI 证明不了 app 能不能跑起来"那句话是对的，
但我当时把它当成了免责声明而不是**待办**。姊妹库 `pll_simulator` 早就有
`android_page_harness.py`，我没照着建一个。

## 口径差异（决定，不是缺口）

| 手机上没有 | 为什么 |
|---|---|
| RTL / AMS 导出 | 输出是 Verilog 目录，验证要 shell 调 `iverilog`；两样在手机上都没有意义 |
| `run_mc_parallel` | 报告层走串行 `run_mc`；`ProcessPoolExecutor` 不该在 Android 下 fork |

页面五个标签覆盖 bridge 全部 7 个方法，双向由 `tests/test_android_parity.py`
卡死。

## 开放问题

- **没有在真机上跑过。** 交叉编译干净、wheel 看着对、CI 全绿，都不等于真机
  `import` 成功；手势手感、刘海安全区同样只有真机看得到。侧载一台做一次真实
  计算之前，"能用"这句话不成立。
- 双抽头 FIR（osr=50）在 x86 runner 上 2.1 s，手机上按几倍估——是否需要给它
  一个更小的手机默认值，等真机数据再定。
