/* Drive the Android page in a real DOM, with a stub bridge.
 *
 * tests/test_android_parity.py checks what can be checked from text: that
 * the bridge and the page name the same methods and ids.  It cannot see
 * what happens once the page RUNS — and the first real device build shipped
 * a page whose Run buttons did nothing at all, because applyLang() set
 * textContent on <label> elements that wrapped the controls and deleted
 * every input from the DOM.  Every static check still passed: the ids were
 * all present in index.html, they just stopped existing a moment later.
 *
 * So this executes app.js against index.html, boots it, and clicks each Run
 * button, asserting the call actually reaches the bridge with usable
 * arguments.  Run by tests/test_android_page.py (node + jsdom).
 */
"use strict";

const fs = require("fs");
const path = require("path");
const { JSDOM, VirtualConsole } = require("jsdom");

// the page lives in the Python package; the APK takes it as its assets
const WWW = path.resolve(__dirname, "../src/polartx/webui/www");

/* What the buttons are expected to send, and which argument proves the
 * form was actually read rather than defaulted away. */
const BUTTONS = [
  { id: "run-compare", method: "compare", key: "seed" },
  { id: "run-fir", method: "fir", key: "bw" },
  { id: "run-combiner", method: "combiner", key: "n_way" },
  { id: "run-selector", method: "selector", key: "bw_hz" },
  { id: "run-mc", method: "montecarlo", key: "n_chips" },
];

/* Minimal stand-in for polartx.appbridge: records what it is asked for and
 * answers in the same envelope the real one uses. */
function makeHost(win, calls) {
  return {
    call(id, method, argsJson) {
      calls.push({ method, args: JSON.parse(argsJson) });
      const result =
        method === "version"
          ? { lib: "0.0.0-test", python: "3.10.0", numpy: "1.26.0" }
          : method === "list_presets"
            ? { presets: ["BLE LE-1M", "WiFi 160 MHz"],
                benchmarks: ["Bench: Degani'24 WiFi7"],
                default: "WiFi 160 MHz",
                groups: { envelope: ["包络通路", "envelope path"] },
                specs: { "WiFi 160 MHz": [{
                  id: "skew_ns", kw: "env_skew_s", kind: "float",
                  zh: "AM/PM skew", en: "AM/PM skew", unit: "ns",
                  group: "envelope", lo: -4, hi: 4, step: 0.05, default: 0,
                  allow_off: false, choices: [], help_zh: "", help_en: "" }] } }
            : { metrics: { "EVM [dB]": -40.0, mask: "PASS" }, png: "",
                metric_info: {
                  "EVM [dB]": { zh: "EVM", en: "EVM", unit: "dB", kind: "number" },
                  mask: { zh: "频谱 mask", en: "mask", unit: "", kind: "verdict" } },
                plots: [{ id: "spectrum", title: "s", series: [
                  { style: "trace", x: [0, 1], y: [0, -1] }] }],
                conventions: [{ id: "mask", zh: "mask", en: "mask" }],
                warnings: [] };
      // the real bridge replies asynchronously, off the binder thread
      setTimeout(
        () => win.onHostReply(id, JSON.stringify({ ok: true, result })), 0);
    },
  };
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function main() {
  const failures = [];
  const virtualConsole = new VirtualConsole();
  const pageErrors = [];
  // jsdom has no canvas; the chart code checks for a context and draws
  // nothing, which is right — the harness is about wiring, not pixels
  virtualConsole.on("jsdomError", e => {
    if (!/HTMLCanvasElement.*getContext/.test(e.message))
      pageErrors.push(e.message);
  });

  const dom = new JSDOM(fs.readFileSync(path.join(WWW, "index.html"), "utf8"), {
    runScripts: "outside-only",
    resources: undefined,
    virtualConsole,
  });
  const win = dom.window;
  const calls = [];
  win.host = makeHost(win, calls);
  // an uncaught error inside a click listener is exactly how a button goes
  // silent, so record rather than ignore
  win.addEventListener("error", e => pageErrors.push(String(e.error || e.message)));

  // every script the page loads, in the page's own order
  const html = fs.readFileSync(path.join(WWW, "index.html"), "utf8");
  for (const m of html.matchAll(/<script src="([^"]+)"/g)) {
    win.eval(fs.readFileSync(path.join(WWW, m[1]), "utf8"));
  }
  await sleep(60);                                   // let boot() settle

  const doc = win.document;
  if (doc.getElementById("app").hidden) {
    failures.push("boot never revealed #app — the page is stuck on the boot card");
  }
  if (!calls.some(c => c.method === "version")) {
    failures.push("boot never called version");
  }

  /* The regression that shipped: controls must survive applyLang(). */
  const controls = ["bench-preset", "bench-seed", "bench-noise", "knob-skew_ns",
                    "fir-bw", "fir-notch", "fir-syms", "fir-osr",
                    "cb-nway", "cb-peaking", "cb-backoff", "cb-loss",
                    "sel-bw", "sel-evm", "sel-fout", "sel-bits",
                    "mc-n", "mc-sigma", "mc-limit", "mc-bw"];
  const gone = controls.filter(id => !doc.getElementById(id));
  if (gone.length) {
    failures.push(`controls destroyed after boot: ${gone.join(", ")}`);
  }

  const sel = doc.getElementById("bench-preset");
  if (sel && sel.options.length === 0) {
    failures.push("the preset select is present but empty — boot's options were lost");
  }

  /* The workbench has no Run button: it runs on boot and on every change.
   * A control generated from the spec must reach the bridge by its id. */
  await sleep(350);                                  // past the debounce
  if (!calls.some(c => c.method === "chain")) {
    failures.push("the workbench never ran the chain after boot");
  }
  const knob = doc.getElementById("knob-skew_ns");
  if (knob) {
    const before = calls.length;
    knob.value = "0.3";
    knob.dispatchEvent(new win.Event("input", { bubbles: true }));
    await sleep(400);
    const sent = calls.slice(before).filter(c => c.method === "chain");
    if (!sent.length) {
      failures.push("moving a generated control sent no chain call");
    } else if (!sent[0].args.values || sent[0].args.values.skew_ns !== 0.3) {
      failures.push(`the generated control sent values=${JSON.stringify(sent[0].args.values)}`);
    }
  }
  if (!doc.querySelector("#bench-cards .mcard")) {
    failures.push("the workbench rendered no metric cards");
  }

  /* Every Run button must reach the bridge. */
  for (const b of BUTTONS) {
    const before = calls.length;
    const btn = doc.getElementById(b.id);
    if (!btn) { failures.push(`#${b.id} missing`); continue; }
    btn.dispatchEvent(new win.MouseEvent("click", { bubbles: true }));
    await sleep(40);
    const made = calls.slice(before).filter(c => c.method === b.method);
    if (!made.length) {
      failures.push(`#${b.id} click sent no ${b.method} call`);
      continue;
    }
    const args = made[0].args;
    if (!(b.key in args) || args[b.key] === null ||
        (typeof args[b.key] === "number" && Number.isNaN(args[b.key]))) {
      failures.push(
        `#${b.id} sent ${b.method} with unusable ${b.key}=${JSON.stringify(args[b.key])}`);
    }
    const out = doc.getElementById(b.id.replace("run-", "") + "-out");
    void out;   // the render target is asserted statically in the parity test
  }

  /* The busy overlay must be released, or the app is dead after one run. */
  if (!doc.getElementById("busy").hidden) {
    failures.push("the busy overlay is still showing after the runs finished");
  }

  /* The language toggle must not break anything either — same mechanism. */
  doc.getElementById("lang").dispatchEvent(
    new win.MouseEvent("click", { bubbles: true }));
  await sleep(10);
  const goneEn = controls.filter(id => !doc.getElementById(id));
  if (goneEn.length) {
    failures.push(`controls destroyed by the language toggle: ${goneEn.join(", ")}`);
  }

  for (const e of pageErrors) failures.push(`uncaught page error: ${e}`);

  if (failures.length) {
    console.error("FAIL");
    for (const f of failures) console.error("  - " + f);
    process.exit(1);
  }
  console.log(`ok — ${calls.length} bridge calls, ${controls.length} controls intact`);
}

main().catch(e => { console.error("harness crashed: " + e.stack); process.exit(2); });
