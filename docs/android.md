# polartx 上手机：Chaquopy + WebView

第四个前端。Streamlit（网页）、PySide6（桌面）、Windows exe 之外，`android/`
把**真的 CPython** 连同 numpy/scipy/matplotlib 一起塞进 APK，UI 是 WebView，
计算全在本地，**零权限**。

一句话架构：所有逻辑在 Python，藏在**一个函数**后面。

```
WebView (app.js) --host.call(id, method, args)--> MainActivity (单线程 executor)
                 <--window.onHostReply(id, json)--  polartx.appbridge.call()
                                                          |
                                                    polartx.guiutil
                                                    （两个桌面 GUI 也调它）
```

Kotlin 只有约 90 行且不该再变：加功能 = 加一个 Python 函数 + 一段 render。
手机因此是同一批数字的**第三个渲染器**，不是第四份实现——264 项测试照旧覆盖它。

---

## 可行性闸门（先做这个，别的都往后排）

Chaquopy 的纯 Python 包直接从 PyPI 装；**带 C 扩展的只能从 Chaquopy 自己的
Android wheel 仓库装**，PyPI 上根本没有 Android wheel。所以 Python 版本不是
选出来的，是被最重的二进制依赖**决定**的。

`polartx` 的传递性 C 扩展依赖（`pip download` 枚举，非纯 `py3-none-any` 的）：

    numpy  scipy  matplotlib  contourpy  fonttools  kiwisolver  pillow

结论：**Python 3.10**，因为它是 Chaquopy 仓库里**有 scipy wheel 的最新版本**
（chaquo/chaquopy#1237）。

这个结论的依据不是推断，是实测：姊妹库 `pll_simulator` 用**完全相同的三个
二进制依赖**在 Chaquopy 3.10 上构建，*Android APK* workflow 跑过 14 次，最近
一次两个 ABI 交叉编译、ELF 校验、两个 Gradle 构建全部通过。

> 本仓的出口代理封了 `chaquo.com`（403），所以**这里跑不了 Gradle 构建**——
> 那是 CI 的事。闸门本身不受影响，因为它靠的是上面那份实测证据。

版本集是**一起动**的，改一个要重查全部四个：

| 项 | 值 | 为什么 |
|---|---|---|
| Chaquopy `version` | `3.10` | 上面的闸门定的 |
| `buildPython` / CI `setup-python` | **同为 3.10** | 没有任何东西校验它；不一致会在很久以后表现为"pip 拒绝这些 wheel" |
| Chaquopy 插件 | 15.0.1 | 覆盖该 Python 的最新一条线；它也把目标 CPython 定死为 `3.10.13-0` |
| AGP | 8.1.4 | Chaquopy 15.0.1 只有**下限** 7.0.0，无上限 |
| Kotlin / Gradle / Java | 1.9.24 / 8.2 / 17 | 依次配套 |
| `compileSdk` / `minSdk` | 34 / 24 | 34 是 Chaquopy 的地板；24 是我们的选择 |
| `abiFilters` | arm64-v8a, x86_64 | 手机 + 模拟器。每个 ABI 一整份 CPython 和全部二进制 wheel |

---

## 构建

```bash
# 解释版
python -m build --sdist --outdir android/app/pysrc .
gradle -p android :app:assembleDebug          # -> android/app/build/outputs/apk/debug/

# 编译版（先删 sdist，否则 pip 仍有纯 Python 版可解析）
rm -f android/app/pysrc/polartx-*.tar.gz
for abi in arm64-v8a x86_64; do
  python packaging/android_wheel.py --package polartx \
    --compile chain,fir,selector,impairments,montecarlo,measured,presets,guiutil,phasemod,dpa,polar,metrics,cal,analysis,waveforms,export \
    --abi "$abi" --ndk "$ANDROID_NDK_HOME"
done
gradle -p android :app:assembleDebug
```

**Python 一改就要重建 sdist。** Gradle 看不穿 tar.gz，陈旧的 sdist 会静默装上
旧代码——这条链上最常见的"我的修改没生效"。

`.github/workflows/android.yml`（手动触发）一次跑出两个 APK 并互相校验。

---

## 编译版到底买到了什么

`.pyc` 在 APK 里跑一遍 `strings` 就能把函数名、行号和**整段 docstring** 还给
任何人。Cython 编译换成机器码。

**在本仓实测**（带对照组——没有对照组的阴性结果什么都不说明）：

| 检查 | 结果 |
|---|---|
| 对照：**未**编译的 `presets.py` 里的散文 | **仍在**（`immune outright` 命中） |
| 已编译 `chain.so` 里的 docstring/注释散文 | **全无**（5 个短语全部落空） |
| 已编译 `chain.so` 里的**字面常量** | **全在**：`2e6`/`50e-9`/`0.15` 各被 8 字节 `struct.pack` 精确命中 1 次 |

所以老实的说法是：**它把读你算法的成本从"解压即读"抬到"反汇编"**。
它不是许可证校验，不是数据保护，也不掩盖 UI 显示的任何东西。具体地说：

- **WebView 资源仍是明文**：`app.js` 把每个 bridge 方法、参数和整个 UI 流程都写着。
- **未编译模块仍是字节码**：`vendor/`（52 个）、`guiqt/`（5 个）、
  `__init__.py`、`appbridge.py`。
- **字面常量随编译存活**，见上表。
- **运行时能取到的，任何能跑这个 app 的人都能取到。**

### 编译集是怎么定的

`--compile` 故意没有"全部"默认值：**你选的集合是一个断言——删掉这些 `.py`
之后测试套件仍然通过**。所以是这么定的，不是猜的：

1. 建 wheel → 装进干净 venv → 磁盘上**没有任何 `.py` 可回退**（确认 `chain.so`
   旁边没有 `chain.py`）→ 跑全量。
2. 结果：**42 / 101 个模块编译**，**242 passed / 12 skipped**。
3. 顺带测掉一个真风险：`presets` 的 `make_waveform` 靠 `inspect.signature`
   分派 burst 长度。Cython 化后签名内省**仍然可用**——这是测出来的。
4. 体积代价：wheel 1810 → 2030 KiB（40 → 42 个模块那一步）。

**故意不编译的**：

- `vendor/` —— 它是带出处头的**可校验改编副本**，可审计正是它的价值；
  而且那不是 polartx 自己的 IP。
- `guiqt/` —— 桌面专用，手机上根本不导入。
- `appbridge.py` —— 它的**整个方法表在 `app.js` 里就是明文**，编译它保护不了
  任何尚未公开的东西（就是技能文档说的"为保护的表象付出构建面积"）。

---

## 交叉编译才会踩到的坑：`cpow` 未声明

第一次真构建就挂在这里，值得记清楚，因为 `--host` 试跑**抓不到它**：

```
export/rtl.c: error: call to undeclared library function 'cpow'
  #define __Pyx_c_pow_double(a, b)  (cpow(a, b))
  note: include the header <complex.h> …
```

Cython 把它无法证明非负的 `x ** y` 降级成 `cpow()`，却不 include
`<complex.h>`。glibc 的头链会把这些声明间接带进来，所以**主机构建永远看不到**；
NDK 的 clang 不会，而且 C99 之后隐式函数声明是**错误**不是警告。

关键是**这不是"排除掉某个坏模块"能解决的**：扫过全部 42 个模块的生成 C，
**13 个**会真的调用 `cpow`（`fir`、`dpa/dpa`、`polar/split`、`waveforms/ble`、
`metrics/*`、`cal/polar_dpd` …），clang 只是死在第一个上。要说话的是工具链。

`packaging/android_wheel.py` 的 `compile_c` 因此带三个承重开关：

| 开关 | 作用 |
|---|---|
| `-include complex.h` | 让声明无条件存在。对照验证过：不加时报的是和 CI 一字不差的那条错误，加上就干净 |
| `-lm` | Android 的 math 库与 libc 分开 |
| `-Wl,--no-undefined`（**仅交叉路径**） | 把"真机 dlopen 才炸"变成 CI 里的链接失败 |

第三个只在传了 `-lpython` 的交叉路径上加。主机构建**故意不链接 libpython**
（CPython 扩展模块的 C-API 由加载它的解释器解析），在那里要求全部符号解析
会直接炸在 Python API 上——这条也是实测撞出来的，不是推演的。

它的价值正好补上文档里"CI 证明不了什么"的一角：`cpow` 这类平台函数如果在目标
API 级别不存在，`-shared` 链接会**默默放过**，然后在手机上 `dlopen` 失败。
现在它会在 CI 里响。

## 手机上刻意**没有**的功能

口径差异是要**记录的决定**，不是留给下一个人去发现的缺口：

| 缺的 | 为什么 |
|---|---|
| RTL / AMS 导出 | 输出是一个 Verilog 目录，验证要 shell 调 `iverilog`。两样在手机上都没有意义 |
| `run_mc_parallel` | 报告层走串行 `run_mc`；`ProcessPoolExecutor` 不该在 Android 下 fork |

页面有五个标签：链路 / FIR 陷波 / 合路 / 选型 / 良率，覆盖 bridge 的全部
7 个方法。`tests/test_android_parity.py` 双向卡死这件事——多一个没人调的方法
或少一个页面调用的方法，都会红。

---

## 实测耗时（x86 runner，手机上按几倍估）

| 操作 | 耗时 |
|---|---|
| 任一预设跑满默认 burst | 0.15 – 0.55 s |
| 架构选型 / 合路 | 0.11 – 0.15 s |
| 蒙卡 30 片 | 0.61 s |
| **双抽头 FIR（osr=50）** | **2.1 s** ← 本页最慢 |

首次调用还要付 numpy+scipy+matplotlib 的导入，手机上数秒——所以页面在第一个
回复到达前一直停在启动卡片上，否则启动看起来就是卡死。

---

## CI 能证明什么，不能证明什么

`android.yml` 证明：两个 ABI 都交叉编译成功；每个 wheel 的 ELF `e_machine`
与 tag 相符；两个 Gradle 构建都成功；**两个 APK 确实不同**
（`inspect_apk.py --pure` / `--native`，因为两次构建共用一个工作区，真实的
失败模式是第二次复用了第一次的 pip 输出，而构建日志对此只字不提）。

那个网关我按本仓规矩弄坏过一次确认它会红——五个方向：该绿的两个绿，
"解释版冒充编译版"、"编译版冒充解释版"、"源码盖住 .so"三个都红。

**CI 证明不了的，也不该拿"已验证"去暗示的**：这个 app 能不能**跑起来**。
（`-Wl,--no-undefined` 把其中"平台符号缺失"这一类挪进了 CI 能证明的范围，
但仅此一类。）
交叉编译干净、wheel 看着对，都不等于在真机上 `import` 成功。手势手感、
刘海和安全区同样只有真机看得到。**侧载装一台，做一次真实计算，再说完成。**
