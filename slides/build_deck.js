const pptxgen = require("pptxgenjs");

const P = {
  navy: "141A3C",      // dominant
  navy2: "1F2752",
  cyan: "16A8C8",      // supporting
  cyanLt: "E4F4F8",
  mag: "D6208A",       // accent
  ink: "1A1D28",
  mute: "5C6570",
  card: "F4F6FB",
  white: "FFFFFF",
  line: "DCE2EC",
};
const H = "Microsoft YaHei";
const B = "Microsoft YaHei";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";           // 13.3 x 7.5
pres.author = "polar_tx";
pres.title = "polar_tx 数字极坐标发射机仿真库";

const W = 13.33, HT = 7.5, M = 0.7;

// ---------- helpers ----------
function title(s, t, opts = {}) {
  s.addText(t, {
    x: M, y: opts.y || 0.42, w: W - 2 * M, h: 0.7,
    fontFace: H, fontSize: 30, bold: true,
    color: opts.color || P.navy, margin: 0, valign: "middle",
  });
}
function sub(s, t, opts = {}) {
  s.addText(t, {
    x: M, y: opts.y || 1.12, w: W - 2 * M, h: 0.38,
    fontFace: B, fontSize: 13.5, color: opts.color || P.mute, margin: 0,
  });
}
function badge(s, n, x, y, col) {
  s.addShape(pres.ShapeType.ellipse, {
    x, y, w: 0.42, h: 0.42, fill: { color: col },
  });
  s.addText(String(n), {
    x, y, w: 0.42, h: 0.42, fontFace: H, fontSize: 15, bold: true,
    color: P.white, align: "center", valign: "middle", margin: 0,
  });
}
function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.09,
    fill: { color: fill || P.card },
    line: { color: P.line, width: 0.75 },
  });
}
function stat(s, x, y, w, val, label, col) {
  s.addText(val, {
    x, y, w, h: 0.72, fontFace: H, fontSize: 34, bold: true,
    color: col, align: "center", valign: "middle", margin: 0,
  });
  s.addText(label, {
    x, y: y + 0.68, w, h: 0.34, fontFace: B, fontSize: 11.5,
    color: P.mute, align: "center", valign: "middle", margin: 0,
  });
}

/* =========== 1. 封面 =========== */
{
  const s = pres.addSlide();
  s.background = { color: P.navy };
  s.addShape(pres.ShapeType.ellipse, {
    x: 10.4, y: -1.5, w: 5.2, h: 5.2,
    fill: { color: P.navy2 },
  });
  s.addShape(pres.ShapeType.ellipse, {
    x: 11.75, y: -0.15, w: 2.5, h: 2.5,
    fill: { color: P.cyan, transparency: 55 },
  });

  s.addText("polar_tx", {
    x: M, y: 1.75, w: 9, h: 1.0, fontFace: H, fontSize: 50, bold: true,
    color: P.white, margin: 0,
  });
  s.addText("数字极坐标发射机（Digital Polar TX）行为级仿真库", {
    x: M, y: 2.75, w: 9.6, h: 0.5, fontFace: H, fontSize: 20,
    color: P.cyanLt, margin: 0,
  });
  s.addText("从架构选型 → 指标预测 → 校准验证 → RTL 交付的全链路仿真平台", {
    x: M, y: 3.3, w: 9.6, h: 0.42, fontFace: B, fontSize: 13.5,
    color: "9FB0CC", margin: 0,
  });

  const bx = [M, M + 3.05, M + 6.1];
  const bv = [["2", "种发射机架构"], ["16", "条预设链路"], ["195", "项自动化测试"]];
  bv.forEach((v, i) => {
    s.addText(v[0], {
      x: bx[i], y: 4.35, w: 2.6, h: 0.85, fontFace: H, fontSize: 42,
      bold: true, color: P.mag, margin: 0,
    });
    s.addText(v[1], {
      x: bx[i], y: 5.2, w: 2.6, h: 0.34, fontFace: B, fontSize: 12.5,
      color: "9FB0CC", margin: 0,
    });
  });

  s.addText("管理层汇报", {
    x: M, y: 6.5, w: 6, h: 0.32, fontFace: B, fontSize: 12,
    color: "8090AC", margin: 0,
  });
  s.addNotes("polar_tx 项目汇报。全库自研 9,870 行，195 项自动化测试全绿，覆盖窄带与宽带两种数字极坐标发射机架构。");
}

/* =========== 2. 为什么做 =========== */
{
  const s = pres.addSlide();
  title(s, "为什么做数字极坐标：效率");
  sub(s, "终端射频功耗由功率放大器主导；OFDM 的高峰均比让线性 PA 长期工作在回退区");

  const items = [
    ["把幅度交给开关式数字 PA", "极坐标分解后，幅度由数字 PA（DPA）以开关方式合成，回退效率远优于线性方案"],
    ["收益必须可量化才能决策", "本库把「省多少电」算到可比较的精度：同一包络、同一 EVM 口径下横向对比"],
    ["代价同样被量化", "带宽扩展、AM/PM 失配、非线性——极坐标的每一项代价都在模型内，不做乐观假设"],
  ];
  items.forEach((it, i) => {
    const y = 1.75 + i * 1.32;
    badge(s, i + 1, M, y + 0.03, i === 2 ? P.mag : P.cyan);
    s.addText(it[0], {
      x: M + 0.62, y, w: 5.5, h: 0.36, fontFace: H, fontSize: 15, bold: true,
      color: P.ink, margin: 0,
    });
    s.addText(it[1], {
      x: M + 0.62, y: y + 0.38, w: 5.5, h: 0.72, fontFace: B, fontSize: 12,
      color: P.mute, margin: 0,
    });
  });

  s.addChart(pres.ChartType.bar, [{
    name: "平均效率",
    labels: ["线性 PA\n(class-B)", "单核数字 PA\n(SCPA)", "双路 Doherty\n合路"],
    values: [33.4, 44.6, 58.8],
  }], {
    x: 6.85, y: 1.62, w: 5.75, h: 3.9,
    barDir: "col", chartColors: [P.navy, P.cyan, P.mag],
    varyColors: true,
    showTitle: true, title: "调制平均效率（同一 OFDM 包络）",
    titleFontFace: H, titleFontSize: 13, titleColor: P.ink,
    showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: '0.0"%"', dataLabelFontFace: B,
    dataLabelFontSize: 12, dataLabelColor: P.ink,
    showLegend: false,
    catAxisLabelFontFace: B, catAxisLabelFontSize: 10.5,
    catAxisLabelColor: P.mute, catGridLine: { style: "none" },
    valAxisLabelFontFace: B, valAxisLabelFontSize: 10,
    valAxisLabelColor: P.mute, valAxisMaxVal: 70,
    valGridLine: { color: P.line, size: 0.75 },
  });

  card(s, 6.85, 5.72, 5.75, 0.86, P.cyanLt);
  s.addText("同一 −40.7 dB EVM 下，Doherty 合路把平均效率再抬升 14 个百分点", {
    x: 7.05, y: 5.72, w: 5.35, h: 0.86, fontFace: H, fontSize: 12.5,
    bold: true, color: P.navy, valign: "middle", margin: 0,
  });
  s.addNotes("效率是数字极坐标存在的理由。三个数字取自同一 OFDM 包络、同一 EVM 口径的实测仿真，可横向比较。");
}

/* =========== 3. 交付范围 =========== */
{
  const s = pres.addSlide();
  title(s, "交付范围：两种架构、六类制式");
  sub(s, "共用一条可组合信号链，只替换相位调制器，便于同口径横向对比");

  const arch = [
    { t: "窄带：ADPLL 两点调制", c: P.navy,
      d: "相位在环内合成，无 DTC 量化/抖动/INL 地板；直通 DAC 覆盖能力约束带宽上限",
      list: ["蓝牙 BLE（LE-1M / LE-2M）", "蓝牙 EDR（π/4-DQPSK、8DPSK）", "EDGE（线性化 GMSK）", "LTE ≤ 20 MHz"] },
    { t: "宽带：开环 DTC 相位调制", c: P.cyan,
      d: "带宽不受环路限制，代价是 DTC 量化、边沿抖动与 INL 三项额外噪声地板",
      list: ["WiFi 6 / 7（≤ 320 MHz）", "WiFi MLO（双抽头 FIR）", "5G NR FR1 / FR2（≤ 200 MHz）", "4096-QAM / MCS13"] },
  ];
  arch.forEach((a, i) => {
    const x = M + i * 6.15;
    card(s, x, 1.72, 5.75, 3.42);
    s.addShape(pres.ShapeType.ellipse, {
      x: x + 0.32, y: 2.02, w: 0.34, h: 0.34, fill: { color: a.c } });
    s.addText(a.t, {
      x: x + 0.82, y: 1.98, w: 4.7, h: 0.42, fontFace: H, fontSize: 15.5,
      bold: true, color: P.ink, margin: 0, valign: "middle" });
    s.addText(a.d, {
      x: x + 0.32, y: 2.52, w: 5.1, h: 0.66, fontFace: B, fontSize: 11.5,
      color: P.mute, margin: 0 });
    s.addText(a.list.map((v, k) => ({
      text: v, options: { bullet: true, breakLine: k !== a.list.length - 1 } })), {
      x: x + 0.36, y: 3.24, w: 5.05, h: 1.72, fontFace: B, fontSize: 12,
      color: P.ink, paraSpaceAfter: 7, margin: 0 });
  });

  const st = [["16", "预设链路"], ["6", "文献级对标"], ["18", "可运行示例"], ["9,870", "行自研代码"]];
  st.forEach((v, i) => {
    const x = M + i * 3.03;
    card(s, x, 5.42, 2.78, 1.18, P.white);
    stat(s, x, 5.5, 2.78, v[0], v[1], i === 1 ? P.mag : P.navy);
  });
  s.addNotes("两条链路共用同一套波形、损伤与指标基础设施，所以窄带与宽带的结果可以直接比较，不存在口径差异。");
}

/* =========== 4. 全链路能力 =========== */
{
  const s = pres.addSlide();
  title(s, "一条信号链，覆盖设计全流程");
  sub(s, "每一级都可注入损伤、可单独观测，指标口径与实测仪器对齐");

  const flow = ["波形生成", "CFR\n削峰", "极坐标\n分解", "相位路径\n包络路径", "数字 PA\n(DPA)", "指标\nEVM/ACLR/Mask"];
  const fw = 1.78, gap = 0.24;
  const x0 = M + 0.1;
  flow.forEach((t, i) => {
    const x = x0 + i * (fw + gap);
    const isEnd = i === flow.length - 1;
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 1.78, w: fw, h: 1.12, rectRadius: 0.08,
      fill: { color: isEnd ? P.mag : (i === 0 ? P.navy : P.cyan) },
    });
    s.addText(t, {
      x, y: 1.78, w: fw, h: 1.12, fontFace: H, fontSize: 12, bold: true,
      color: P.white, align: "center", valign: "middle", margin: 0,
    });
    if (i < flow.length - 1) {
      s.addText("▶", {
        x: x + fw, y: 1.78, w: gap, h: 1.12, fontFace: B, fontSize: 11,
        color: P.mute, align: "center", valign: "middle", margin: 0 });
    }
  });

  const caps = [
    ["损伤注入", "AM/PM 时延失配、两点增益误差、DTC 量化与 INL、供电推压、单元失配"],
    ["校准闭环", "时延估计（波形对齐 / ACP 功率搜索）、两点增益 LS 与在线 LMS、DTC LUT 标定"],
    ["预失真", "极坐标双 LUT（AM-AM 逆 + AM-PM 补偿）、整链 ILA-GMP 记忆 DPD"],
    ["良率与交付", "Monte Carlo 逐芯片抽样、RTL + Verilog-AMS 导出（iverilog 逐位验证）"],
  ];
  caps.forEach((c, i) => {
    const x = M + (i % 2) * 6.15;
    const y = 3.28 + Math.floor(i / 2) * 1.72;
    card(s, x, y, 5.75, 1.5);
    badge(s, i + 1, x + 0.3, y + 0.28, i === 3 ? P.mag : P.navy);
    s.addText(c[0], {
      x: x + 0.85, y: y + 0.24, w: 4.6, h: 0.36, fontFace: H, fontSize: 14,
      bold: true, color: P.ink, margin: 0, valign: "middle" });
    s.addText(c[1], {
      x: x + 0.85, y: y + 0.66, w: 4.65, h: 0.68, fontFace: B, fontSize: 11.5,
      color: P.mute, margin: 0 });
  });
  s.addNotes("这条链不是黑盒：每一级的中间量都保留，可单独观察，也可单独注入损伤做灵敏度分析。");
}

/* =========== 5. 可信度 =========== */
{
  const s = pres.addSlide();
  title(s, "结果可信度：对标 2005–2026 六篇文献");
  sub(s, "不是自证——每条链路都对着公开发表的流片结果核对，落在同一量级内");

  const rows = [
    ["Staszewski, JSSC 2005", "EDGE 极坐标", "DEVM 2–3 %", "2.18 %"],
    ["802.11n 世代（约 2010）", "WLAN 20 MHz / 64-QAM", "≈ −28 dB", "−28.0 dB"],
    ["Madoglio, ISSCC 2014", "LTE 20 MHz", "≈ −30 dB", "−31.0 dB"],
    ["Ben Bassat, JSSC 2020", "WiFi 6 / 160 MHz", "−38 dB", "−38.1 dB"],
    ["Degani, RFIC 2024", "WiFi 7 / 320 MHz", "−38 dB 档", "−37.0 dB"],
    ["Borokhovich, RFIC 2026", "WiFi MLO / FIR+Doherty", "−40.7 dB", "−42.7 dB"],
  ];
  const tb = [[
    { text: "文献", options: { bold: true, color: P.white, fill: { color: P.navy }, fontFace: H } },
    { text: "制式", options: { bold: true, color: P.white, fill: { color: P.navy }, fontFace: H } },
    { text: "发表值", options: { bold: true, color: P.white, fill: { color: P.navy }, fontFace: H } },
    { text: "本库仿真", options: { bold: true, color: P.white, fill: { color: P.mag }, fontFace: H } },
  ]];
  rows.forEach((r, i) => {
    const bg = i % 2 ? P.card : P.white;
    tb.push([
      { text: r[0], options: { fill: { color: bg }, color: P.ink } },
      { text: r[1], options: { fill: { color: bg }, color: P.mute } },
      { text: r[2], options: { fill: { color: bg }, color: P.mute } },
      { text: r[3], options: { fill: { color: P.cyanLt }, color: P.navy, bold: true } },
    ]);
  });
  s.addTable(tb, {
    x: M, y: 1.75, w: W - 2 * M, colW: [3.5, 3.2, 2.4, 2.83],
    rowH: 0.415, fontFace: B, fontSize: 11.5, valign: "middle",
    border: { type: "solid", color: P.line, pt: 0.75 },
    margin: [4, 8, 4, 8],
  });

  const q = [
    ["195 项", "自动化测试（2 项因缺数据集跳过）"],
    ["9 个", "CI 作业，Windows/macOS/Linux 全绿"],
    ["物理量断言", "断言量化噪底、失配 √N 律等物理规律，非快照比对"],
  ];
  q.forEach((v, i) => {
    const x = M + i * 4.05;
    card(s, x, 4.92, 3.75, 1.46, P.white);
    s.addText(v[0], {
      x: x + 0.25, y: 5.08, w: 3.3, h: 0.5, fontFace: H, fontSize: 21,
      bold: true, color: i === 2 ? P.mag : P.cyan, margin: 0 });
    s.addText(v[1], {
      x: x + 0.25, y: 5.6, w: 3.3, h: 0.68, fontFace: B, fontSize: 11.5,
      color: P.mute, margin: 0 });
  });
  s.addNotes("对标覆盖 20 年代际跨度，从 EDGE 到最新的 WiFi MLO。差异都在可解释范围内，且每条对标都有测试锁定，防止后续改动悄悄偏离。");
}

/* =========== 6. 关键发现 =========== */
{
  const s = pres.addSlide();
  title(s, "代表性技术发现");
  sub(s, "这些结论直接影响架构选型与版图/校准资源的分配");

  const f = [
    ["AM/PM 时延失配是最尖锐的损伤", P.mag,
      "WiFi 160 MHz 下 0.2 ns 即导致频谱模板失败。已内建两种估计器（波形对齐 / 仅用功率检波的 ACP 搜索），校准后可回到基线。"],
    ["ACP 比 EVM 更早报警", P.navy,
      "频谱仪没有均衡器，功率域完全暴露损伤；而 EVM 会被接收机均衡「洗白」。研发验证应以 ACP/模板为先。"],
    ["效率优势由物理推导，非曲线拟合", P.cyan,
      "Doherty 合路从负载调制推出效率与 AM-AM/AM-PM，核间失配可做良率蒙卡，结论可外推而不只是拟合当前数据。"],
    ["实测硅数据可直接入链", P.navy,
      "接入公开实测 DPA 数据集：静态极坐标模型与含记忆模型相差约 19 dB，该差距量化了器件记忆效应的代价。"],
  ];
  f.forEach((c, i) => {
    const x = M + (i % 2) * 6.15;
    const y = 1.78 + Math.floor(i / 2) * 2.42;
    card(s, x, y, 5.75, 2.18);
    s.addShape(pres.ShapeType.ellipse, {
      x: x + 0.32, y: y + 0.3, w: 0.36, h: 0.36, fill: { color: c[1] } });
    s.addText(String(i + 1), {
      x: x + 0.32, y: y + 0.3, w: 0.36, h: 0.36, fontFace: H, fontSize: 14,
      bold: true, color: P.white, align: "center", valign: "middle", margin: 0 });
    s.addText(c[0], {
      x: x + 0.86, y: y + 0.26, w: 4.66, h: 0.44, fontFace: H, fontSize: 14.5,
      bold: true, color: P.ink, margin: 0, valign: "middle" });
    s.addText(c[2], {
      x: x + 0.36, y: y + 0.82, w: 5.1, h: 1.18, fontFace: B, fontSize: 11.5,
      color: P.mute, margin: 0 });
  });
  s.addNotes("第 1、2 条是验证策略层面的结论：先看 ACP 和模板，再看 EVM。第 3 条说明效率模型可外推。第 4 条说明模型与真实硅之间的差距是已知且量化的。");
}

/* =========== 7. 工程化 =========== */
{
  const s = pres.addSlide();
  title(s, "工程化程度：可交付、可复现、可维护");
  sub(s, "不是一次性脚本，而是有测试、有 CI、有打包、可长期维护的工具");

  const nums = [["9,870", "行自研代码"], ["195", "项自动化测试"],
                ["39", "文件测试套件"], ["18", "个可运行示例"]];
  nums.forEach((v, i) => {
    const y = 1.78 + i * 1.22;
    card(s, M, y, 5.75, 1.06, P.white);
    s.addText(v[0], {
      x: M + 0.3, y, w: 2.1, h: 1.06, fontFace: H, fontSize: 26, bold: true,
      color: i === 1 ? P.mag : P.navy, valign: "middle", margin: 0 });
    s.addText(v[1], {
      x: M + 2.4, y, w: 3.1, h: 1.06, fontFace: B, fontSize: 13,
      color: P.mute, valign: "middle", margin: 0 });
  });

  const d = [
    ["双 GUI 工作台", "网页版（Streamlit）与桌面版（PySide6），六个功能页，非工程师亦可操作"],
    ["Windows 单文件 exe", "四种构建组合，免安装分发给验证与系统团队"],
    ["RTL + Verilog-AMS 导出", "数字数据通路可综合，模拟 PA 实数模型可协仿，全部逐位对齐 Python 金向量"],
    ["Vendor 漂移检查", "自动核对与两个姊妹库的同步状态，防止长期演进中悄悄脱节"],
  ];
  d.forEach((c, i) => {
    const y = 1.78 + i * 1.22;
    card(s, M + 6.15, y, 5.75, 1.06);
    s.addShape(pres.ShapeType.ellipse, {
      x: M + 6.4, y: y + 0.33, w: 0.4, h: 0.4,
      fill: { color: i === 3 ? P.mag : P.cyan } });
    s.addText(c[0], {
      x: M + 6.95, y: y + 0.13, w: 4.7, h: 0.34, fontFace: H, fontSize: 13,
      bold: true, color: P.ink, margin: 0, valign: "middle" });
    s.addText(c[1], {
      x: M + 6.95, y: y + 0.47, w: 4.75, h: 0.5, fontFace: B, fontSize: 10.5,
      color: P.mute, margin: 0 });
  });
  s.addNotes("重点是可维护性：新人改动会被 195 项测试和多平台 CI 挡住，vendor 漂移检查保证与姊妹库不脱节。");
}

/* =========== 8. 结论 =========== */
{
  const s = pres.addSlide();
  s.background = { color: P.navy };
  s.addShape(pres.ShapeType.ellipse, {
    x: -1.6, y: 4.6, w: 4.6, h: 4.6, fill: { color: P.navy2 } });

  title(s, "结论与后续", { color: P.white });
  s.addText("平台已具备支撑架构决策的完整能力", {
    x: M, y: 1.12, w: 11, h: 0.4, fontFace: B, fontSize: 14,
    color: "9FB0CC", margin: 0 });

  const c = [
    ["能力完整", "架构选型 → 指标预测 → 校准验证 → RTL 交付，四个环节全部打通并有示例支撑"],
    ["结果可信", "六篇文献对标 + 195 项物理量断言测试 + 多平台 CI，结论不依赖单次运行"],
    ["可直接使用", "双 GUI 与免安装 exe 已就绪，系统与验证团队无需环境配置即可上手"],
  ];
  c.forEach((v, i) => {
    const x = M + i * 4.05;
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 1.85, w: 3.75, h: 2.35, rectRadius: 0.09,
      fill: { color: P.navy2 }, line: { color: "34406E", width: 0.75 } });
    s.addShape(pres.ShapeType.ellipse, {
      x: x + 0.32, y: 2.15, w: 0.4, h: 0.4,
      fill: { color: i === 2 ? P.mag : P.cyan } });
    s.addText(v[0], {
      x: x + 0.32, y: 2.68, w: 3.1, h: 0.42, fontFace: H, fontSize: 17,
      bold: true, color: P.white, margin: 0 });
    s.addText(v[1], {
      x: x + 0.32, y: 3.12, w: 3.15, h: 0.96, fontFace: B, fontSize: 11.5,
      color: "A9B8D4", margin: 0 });
  });

  s.addText("后续可选方向", {
    x: M, y: 4.62, w: 6, h: 0.36, fontFace: H, fontSize: 15, bold: true,
    color: P.white, margin: 0 });
  const nx = ["扩展制式覆盖（WiFi 8、NR-U 等新 numerology）",
              "回灌自有流片实测数据，校准模型与硅的差距",
              "延伸到系统级功耗与热建模，支撑整机功耗预算"];
  s.addText(nx.map((t, i) => ({
    text: t, options: { bullet: true, breakLine: i !== nx.length - 1 } })), {
    x: M, y: 5.06, w: 11.6, h: 1.3, fontFace: B, fontSize: 12.5,
    color: "A9B8D4", paraSpaceAfter: 8, margin: 0 });
  s.addNotes("后续方向按需展开；当前平台已可直接投入使用，不存在阻塞项。");
}

pres.writeFile({ fileName: "/home/user/polar_tx/slides/polar_tx_管理层汇报.pptx" })
  .then(f => console.log("written:", f));
