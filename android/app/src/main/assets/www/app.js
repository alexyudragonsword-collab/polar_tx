/* The polartx phone UI, over the polartx.appbridge JSON RPC.
 *
 * Plain HTML/JS, no build step and no framework: the app's complexity
 * belongs in Python where the 238-test suite already covers it, and this
 * file should stay boring enough to read in one sitting.
 *
 * Every call() name here must exist in appbridge._METHODS and every method
 * there must be called from here — tests/test_android_parity.py checks both
 * directions from text alone, because a bridge method with no caller looks
 * tested and does nothing.
 */
"use strict";

/* ---------------------------------------------------------- host RPC */
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
  if (on) $("busy-text").textContent = lang === "zh" ? textZh : textEn;
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

/* One runner for every tab: same busy handling, same single error branch.
 * The overlay is released in `finally` — an early return that skips it
 * leaves the app permanently unresponsive with nothing on screen to say so. */
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

/* ------------------------------------------------------------- tabs */
document.querySelectorAll("#tabs button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#tabs button").forEach(b =>
      b.classList.toggle("active", b === btn));
    document.querySelectorAll(".tab").forEach(t =>
      t.hidden = t.id !== "tab-" + btn.dataset.tab);
  });
});

$("lang").addEventListener("click", () => {
  lang = lang === "zh" ? "en" : "zh";
  applyLang();
});

/* ------------------------------------------------------------ chain */
$("run-chain").addEventListener("click", () => run(
  "ch-out", "chain", {
    preset: $("ch-preset").value,
    seed: parseInt($("ch-seed").value, 10),
    noise: $("ch-noise").checked,
    env_skew_ns: num("ch-skew"),
  },
  "正在跑链路…", "running the chain…"));

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
    const sel = $("ch-preset");
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
    $("boot").hidden = true;
    $("app").hidden = false;
    applyLang();
  } catch (e) {
    $("boot-error").hidden = false;
    $("boot-error").textContent = String(e.message || e);
  }
})();
