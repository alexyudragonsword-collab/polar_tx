/* The polartx page, over the polartx.appbridge JSON RPC.
 *
 * One page, three hosts: the Android WebView injects `window.host`; on a
 * desktop or in a browser, polartx.webhost serves this directory and the
 * shim below talks to it over HTTP.  Either way every number comes from
 * Python (guiutil) and every control is described by Python (guispec):
 * this file lays out what it is given and computes no physics.
 *
 * Plain HTML/JS, no build step and no framework, kept boring enough to read
 * in one sitting.  Every call() name here must exist in appbridge._METHODS
 * and every method there must be called from here —
 * tests/test_android_parity.py checks both directions from text alone,
 * because a bridge method with no caller looks tested and does nothing.
 */
"use strict";

/* ---------------------------------------------------------- host RPC */
/* Desktop / browser: no injected interface, so reach the local server. */
if (!window.host) {
  window.host = {
    call(id, method, argsJson) {
      fetch("call", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ method, args: JSON.parse(argsJson) }),
      }).then(r => r.text())
        .then(t => window.onHostReply(id, t))
        .catch(e => window.onHostReply(id, JSON.stringify(
          { ok: false, error: "host unreachable: " + e })));
    },
  };
}

const pending = {};
let seq = 0;

/* Async by construction: @JavascriptInterface methods arrive on a binder
 * thread, and blocking one while Python computes freezes the UI for the
 * length of the call.  host.call returns immediately; the answer arrives
 * through window.onHostReply. */
function call(method, args) {
  return new Promise((resolve, reject) => {
    const id = String(++seq);
    pending[id] = { resolve, reject };
    window.host.call(id, method, JSON.stringify(args || {}));
  });
}

window.onHostReply = (id, replyStr) => {
  const p = pending[id];
  delete pending[id];
  if (!p) return;
  let r;
  try { r = JSON.parse(replyStr); }
  catch (e) { p.reject(new Error("bad reply: " + e)); return; }
  if (r.ok) p.resolve(r.result);
  else p.reject(new Error(r.error));
};

/* ---------------------------------------------------------- language */
let lang = "zh";
const tr = (zh, en) => (lang === "zh" ? zh : en);

/* Swap every translatable string.  This assigns textContent, which REPLACES
 * ALL CHILDREN — so an element carrying data-zh must never wrap another
 * element.  Put the label text in an inner <span> and tag that instead:
 *
 *     <label><span data-zh="种子" data-en="Seed">种子</span><input …></label>
 *
 * The first device build got this wrong: 21 <label data-zh=…> elements
 * wrapped their own <input>, so this line deleted every control from the DOM
 * the moment boot() finished, and every Run button then threw on a null and
 * did nothing visible at all.  tests/test_android_parity.py enforces the
 * invariant statically; tests/android_page_harness.js catches it at runtime. */
function applyLang() {
  document.querySelectorAll("[data-zh]").forEach(el => {
    el.textContent = el.dataset[lang];
  });
  document.getElementById("lang").textContent = lang === "zh" ? "EN" : "中文";
  document.documentElement.lang = lang;
}

/* ---------------------------------------------------------- helpers */
const $ = id => document.getElementById(id);
const num = id => parseFloat($(id).value);
const esc = s => String(s).replace(/[&<>"]/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function busy(on, textZh, textEn) {
  $("busy").hidden = !on;
  if (on) $("busy-text").textContent = tr(textZh, textEn);
  document.querySelectorAll("button.primary").forEach(b => b.disabled = on);
}

/* PASS/FAIL are the two values worth colouring; everything else is a
 * number the reader interprets. */
function metricsHtml(metrics) {
  return '<div class="metrics">' + Object.entries(metrics).map(([k, v]) => {
    const s = v === null ? "n/a" : String(v);
    const cls = s === "PASS" ? " pass" : s === "FAIL" ? " fail" : "";
    return `<div class="metric${cls}"><b>${esc(s)}</b><span>${esc(k)}</span></div>`;
  }).join("") + "</div>";
}

function pngHtml(b64) {
  return b64 ? `<img class="plot" src="data:image/png;base64,${b64}">` : "";
}

function errHtml(e) {
  return `<p class="error">${esc(e.message || e)}</p>`;
}

/* One runner for the slow tabs: same busy handling, same single error
 * branch.  The overlay is released in `finally` — an early return that
 * skips it leaves the app permanently unresponsive with nothing on screen
 * to say so. */
async function run(outId, method, args, textZh, textEn, render) {
  busy(true, textZh, textEn);
  try {
    const r = await call(method, args);
    $(outId).innerHTML = render ? render(r)
      : metricsHtml(r.metrics || r.summary || {}) + pngHtml(r.png);
  } catch (e) {
    $(outId).innerHTML = errHtml(e);
  } finally {
    busy(false);
  }
}

/* -------------------------------------------------------- workbench */
/* The workbench recomputes on every change.  A chain run takes 0.1-0.7 s
 * on a laptop (a few seconds on a phone), so requests are coalesced: one
 * in flight, and when it lands the latest settings run once more if they
 * moved meanwhile — the page never queues a backlog of stale runs. */
const B = {
  specs: {}, groups: {}, preset: "", values: {},   // values: knob id -> display value
  result: null, panel: 0, ms: 0,
  inflight: false, again: false, timer: 0,
};
const specsOf = name => B.specs[name] || [];
const same = (a, b) => a === b || (typeof a === "number" && typeof b === "number"
                                   && Math.abs(a - b) < 1e-9);

function benchStatus(text) { $("bench-status").textContent = text; }

async function runBench() {
  if (B.inflight) { B.again = true; return; }
  B.inflight = true;
  B.again = false;
  const t0 = performance.now();
  benchStatus(tr("计算中…", "computing…"));
  try {
    const r = await call("chain", {
      preset: B.preset, seed: parseInt($("bench-seed").value, 10) || 0,
      noise: $("bench-noise").checked, values: B.values, png: false,
    });
    B.result = r;
    B.ms = performance.now() - t0;
    renderBench();
  } catch (e) {
    $("bench-notes").innerHTML = errHtml(e);
    benchStatus("");
  } finally {
    B.inflight = false;
    if (B.again) runBench();
    else $("tab-bench").classList.remove("stale");
  }
}

function schedule() {
  clearTimeout(B.timer);
  $("tab-bench").classList.add("stale");
  B.timer = setTimeout(runBench, 250);
}

function valueText(k, v) {
  if (v === "off") return tr("关", "off");
  if (k.kind === "bool") return v ? tr("开", "on") : tr("关", "off");
  if (k.kind === "choice") return String(v);
  return `${Charts.fmt(v)}${k.unit ? " " + k.unit : ""}`;
}

/* One control from its guispec description.  Labels, unit, range and the
 * preset's own default all come from Python; nothing here is typed in. */
function knobEl(k) {
  const wrap = document.createElement("div");
  wrap.className = "knob";
  const cur = () => (k.id in B.values ? B.values[k.id] : k.default);
  const head = document.createElement("div");
  head.className = "head";
  const name = document.createElement("span");
  const val = document.createElement("span");
  val.className = "val";
  head.append(name, val);
  const refresh = () => {
    name.innerHTML = esc(tr(k.zh, k.en))
      + (same(cur(), k.default) ? "" : '<span class="dot"></span>');
    val.textContent = valueText(k, cur());
  };
  const set = v => {
    if (same(v, k.default)) delete B.values[k.id];
    else B.values[k.id] = v;
    refresh();
    schedule();
  };
  wrap.appendChild(head);

  if (k.kind === "bool") {
    const lab = document.createElement("label");
    lab.className = "check";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.id = "knob-" + k.id;
    box.checked = !!cur();
    box.addEventListener("change", () => set(box.checked));
    const t = document.createElement("span");
    t.textContent = tr("启用", "enabled");
    lab.append(box, t);
    wrap.appendChild(lab);
  } else if (k.kind === "choice") {
    const sel = document.createElement("select");
    sel.id = "knob-" + k.id;
    for (const [v, zh, en] of k.choices) sel.appendChild(new Option(tr(zh, en), v));
    sel.value = cur();
    sel.addEventListener("change", () => set(sel.value));
    wrap.appendChild(sel);
  } else {
    const range = document.createElement("input");
    range.type = "range";
    range.id = "knob-" + k.id;
    range.min = k.lo; range.max = k.hi; range.step = k.step;
    const start = cur() === "off" ? (k.lo + k.hi) / 2 : cur();
    range.value = start;
    const parse = () => (k.kind === "int" ? parseInt(range.value, 10)
                                         : parseFloat(range.value));
    range.addEventListener("input", () => set(parse()));
    wrap.appendChild(range);
    if (k.allow_off) {
      const lab = document.createElement("label");
      lab.className = "off";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = cur() === "off";
      range.disabled = box.checked;
      box.addEventListener("change", () => {
        range.disabled = box.checked;
        set(box.checked ? "off" : parse());
      });
      const t = document.createElement("span");
      t.textContent = tr("关闭", "off");
      lab.append(box, t);
      wrap.appendChild(lab);
    }
  }
  const help = tr(k.help_zh, k.help_en);
  if (help) {
    const p = document.createElement("div");
    p.className = "help";
    p.textContent = help;
    wrap.appendChild(p);
  }
  refresh();
  return wrap;
}

function buildForm() {
  const form = $("bench-form");
  form.textContent = "";
  const specs = specsOf(B.preset);
  if (!specs.length) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = tr("文献对标：参数固定为发表时的假设（见预设 docstring）",
                       "literature benchmark: parameters fixed to the "
                       + "published-class assumptions (see the preset docstring)");
    form.appendChild(p);
    return;
  }
  for (const [gid, names] of Object.entries(B.groups)) {
    const ks = specs.filter(k => k.group === gid);
    if (!ks.length) continue;
    const h = document.createElement("div");
    h.className = "kgroup";
    h.textContent = tr(names[0], names[1]);
    form.appendChild(h);
    ks.forEach(k => form.appendChild(knobEl(k)));
  }
}

const VERDICT_ZH = { PASS: "通过", FAIL: "失败", resolved: "可分辨",
                     marginal: "边缘", "clouds overlap at this EVM": "云团重叠" };

function cardHtml(key, v, info) {
  const label = esc(info ? info[lang] : key);
  if (info && info.kind === "verdict") {
    const cls = v === "PASS" ? "pass" : "fail";
    return `<div class="mcard"><div class="k"><span>${label}</span>`
      + `<span class="pill ${cls}">${esc(tr(VERDICT_ZH[v] || v, v))}</span></div></div>`;
  }
  if (info && info.kind === "text") {
    const s = String(v);
    const head = s.split(" (")[0];
    const zh = (VERDICT_ZH[head] || head) + s.slice(head.length);
    return `<div class="mcard text"><div class="k"><span>${label}</span></div>`
      + `<div class="v">${esc(tr(zh, s))}</div></div>`;
  }
  const unit = info && info.unit ? ` <small>${esc(info.unit)}</small>` : "";
  const shown = typeof v === "number" ? Charts.fmt(v) : (v === null ? "n/a" : v);
  return `<div class="mcard"><div class="k"><span>${label}</span></div>`
    + `<div class="v">${esc(shown)}${unit}</div></div>`;
}

function currentPanel() {
  const plots = (B.result && B.result.plots) || [];
  return plots[Math.min(B.panel, plots.length - 1)];
}

function drawBench(cursor) {
  Charts.draw($("bench-canvas"), currentPanel(), { cursor });
}

function renderBench() {
  const r = B.result;
  if (!r) return;
  const info = r.metric_info || {};
  $("bench-notes").innerHTML = (r.warnings || [])
    .map(w => `<p class="note">⚠ ${esc(w)}</p>`).join("");
  $("bench-cards").innerHTML = Object.entries(r.metrics || {})
    .map(([k, v]) => cardHtml(k, v, info[k])).join("");
  $("bench-conv").innerHTML = (r.conventions || [])
    .map(c => `<span>${esc(c[lang] || c.en)}</span>`).join("");
  const tabs = $("bench-plot-tabs");
  tabs.textContent = "";
  (r.plots || []).forEach((p, i) => {
    const b = document.createElement("button");
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", String(i === B.panel));
    b.textContent = tr(p.title_zh || p.title, p.title).split("\n")[0];
    b.addEventListener("click", () => { B.panel = i; renderBench(); });
    tabs.appendChild(b);
  });
  drawBench();
  benchStatus(`${(B.ms / 1000).toFixed(2)} s`);
}

/* ------------------------------------------------------------- tabs */
function showTab(name) {
  document.querySelectorAll("#nav button").forEach(b =>
    b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach(t =>
    t.hidden = t.id !== "tab-" + name);
  $("nav").classList.remove("open");
  $("nav-toggle").setAttribute("aria-expanded", "false");
  if (name === "bench") drawBench();      // its width was 0 while hidden
}
document.querySelectorAll("#nav button").forEach(btn =>
  btn.addEventListener("click", () => showTab(btn.dataset.tab)));

$("nav-toggle").addEventListener("click", () => {
  const open = $("nav").classList.toggle("open");
  $("nav-toggle").setAttribute("aria-expanded", String(open));
});

$("lang").addEventListener("click", () => {
  lang = lang === "zh" ? "en" : "zh";
  applyLang();
  buildForm();
  renderBench();
});

$("bench-preset").addEventListener("change", () => {
  B.preset = $("bench-preset").value;
  B.values = {};
  B.panel = 0;
  buildForm();
  schedule();
});
$("bench-seed").addEventListener("change", schedule);
$("bench-noise").addEventListener("change", schedule);
$("bench-reset").addEventListener("click", () => {
  B.values = {};
  buildForm();
  schedule();
});
$("bench-sheet").addEventListener("click", () => {
  const c = $("bench-knobs").classList.toggle("collapsed");
  $("bench-sheet").setAttribute("aria-expanded", String(!c));
});

/* the readout: the value under the pointer, read from the report's data */
$("bench-canvas").addEventListener("pointermove", e => {
  const p = Charts.pick($("bench-canvas"), currentPanel(), e.clientX, e.clientY);
  $("bench-readout").textContent = p ? `${Charts.fmt(p.x)}, ${Charts.fmt(p.y)}` : "";
  drawBench(p || undefined);
});
$("bench-canvas").addEventListener("pointerleave", () => {
  $("bench-readout").textContent = "";
  drawBench();
});
let resizeFrame = 0;
window.addEventListener("resize", () => {
  cancelAnimationFrame(resizeFrame);
  resizeFrame = requestAnimationFrame(() => drawBench());
});
if (window.matchMedia) {
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  if (mq.addEventListener) mq.addEventListener("change", () => drawBench());
}

/* -------------------------------------------------------------- fir */
$("run-fir").addEventListener("click", () => run(
  "fir-out", "fir", {
    bw: num("fir-bw") * 1e6,
    notch_offset_hz: num("fir-notch") * 1e6,
    n_symbols: parseInt($("fir-syms").value, 10),
    osr: parseInt($("fir-osr").value, 10),
  },
  "正在跑双抽头 FIR（较慢）…", "running the dual-tap FIR (slow)…"));

/* --------------------------------------------------------- combiner */
$("run-combiner").addEventListener("click", () => run(
  "cb-out", "combiner", {
    n_way: parseInt($("cb-nway").value, 10),
    peaking: $("cb-peaking").value,
    backoff_db: num("cb-backoff"),
    combiner_loss_db: num("cb-loss"),
    gain_imbalance_pct: num("cb-gain"),
    phase_imbalance_deg: num("cb-phase"),
  },
  "正在算合路…", "computing the combiner…"));

/* --------------------------------------------------------- selector */
$("run-selector").addEventListener("click", () => run(
  "sel-out", "selector", {
    bw_hz: num("sel-bw") * 1e6,
    evm_db_max: num("sel-evm"),
    fout: num("sel-fout") * 1e9,
    dtc_bits: parseInt($("sel-bits").value, 10),
    constant_envelope: $("sel-ce").checked,
  },
  "正在打分…", "scoring…",
  r => metricsHtml(r.metrics || {})
     + (r.table ? `<pre class="table">${esc(r.table)}</pre>` : "")
     + pngHtml(r.png)));

/* --------------------------------------------------------------- mc */
$("run-mc").addEventListener("click", () => run(
  "mc-out", "montecarlo", {
    n_chips: parseInt($("mc-n").value, 10),
    bw: num("mc-bw") * 1e6,
    skew_sigma_ns: num("mc-sigma"),
    calibrated: $("mc-cal").checked,
    limit_db: num("mc-limit"),
  },
  "正在抽样…", "sampling chips…"));

/* ------------------------------------------------------------- boot */
/* The first call pays the import of numpy/scipy/matplotlib — several
 * seconds on a phone.  Until it lands the app stays behind the boot card,
 * or the launch looks like a hang. */
(async function boot() {
  try {
    const v = await call("version", {});
    $("version").textContent = `v${v.lib} · py${v.python} · numpy ${v.numpy}`;
    const p = await call("list_presets", {});
    const sel = $("bench-preset");
    for (const [label, names] of [["", p.presets], ["— 对标 / benchmarks —",
                                                    p.benchmarks]]) {
      if (label) {
        const og = document.createElement("optgroup");
        og.label = label;
        names.forEach(n => og.appendChild(new Option(n, n)));
        sel.appendChild(og);
      } else {
        names.forEach(n => sel.appendChild(new Option(n, n)));
      }
    }
    B.specs = p.specs || {};
    B.groups = p.groups || {};
    if (p.default) sel.value = p.default;
    B.preset = sel.value;
    $("boot").hidden = true;
    $("app").hidden = false;
    applyLang();
    buildForm();
    runBench();
  } catch (e) {
    $("boot-error").hidden = false;
    $("boot-error").textContent = String(e.message || e);
  }
})();
