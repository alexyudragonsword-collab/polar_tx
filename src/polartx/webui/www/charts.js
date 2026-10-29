/* Draws a report panel (guispec plot data) on a canvas.
 *
 * It draws exactly the arrays it is handed — the same arrays the desktop
 * PNG is drawn from — and computes nothing about the physics.  Styles:
 * "trace" and "line" (the measurement), "mask" (the limit, dashed),
 * "points" (a constellation), "ref" (a marker line).  Colours come from the
 * page's CSS tokens, so the chart follows light and dark like the rest.
 */
"use strict";
(function () {
  const css = name =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  /* 1-2-5 tick step for a span split into about n intervals */
  function niceStep(span, n) {
    const raw = span / Math.max(1, n);
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const m = raw / p;
    return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
  }

  function fmt(v) {
    if (!isFinite(v)) return "—";
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
    return String(+v.toPrecision(4));
  }

  function extent(panel, axis) {
    const lim = panel[axis + "lim"];
    if (lim) return [lim[0], lim[1]];
    let lo = Infinity, hi = -Infinity;
    for (const s of panel.series) {
      for (const v of s[axis]) {
        if (v === null || !isFinite(v)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    if (!isFinite(lo)) return [0, 1];
    if (lo === hi) { lo -= 1; hi += 1; }
    const pad = axis === "y" ? 0.06 * (hi - lo) : 0;
    return [lo - pad, hi + pad];
  }

  function draw(canvas, panel, opts) {
    opts = opts || {};
    const g = canvas.getContext && canvas.getContext("2d");
    if (!g || !panel) return;               // jsdom, or nothing to draw yet
    const box = canvas.parentElement.clientWidth - 12;
    const w = Math.max(240, box);
    const h = panel.equal ? Math.min(w, 440) : Math.max(220, Math.min(420, w * 0.56));
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.height = h + "px";
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    const small = w < 420;
    const font = (small ? 10 : 11) + "px " + css("--mono");
    const m = { l: small ? 38 : 46, r: 10, t: 10, b: small ? 30 : 34 };
    let [x0, x1] = extent(panel, "x");
    let [y0, y1] = extent(panel, "y");
    const pw = w - m.l - m.r, ph = h - m.t - m.b;
    if (panel.equal) {                      // same scale on both axes
      const sx = (x1 - x0) / pw, sy = (y1 - y0) / ph, s = Math.max(sx, sy);
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      x0 = cx - s * pw / 2; x1 = cx + s * pw / 2;
      y0 = cy - s * ph / 2; y1 = cy + s * ph / 2;
    }
    const X = v => m.l + (v - x0) / (x1 - x0) * pw;
    const Y = v => m.t + (y1 - v) / (y1 - y0) * ph;

    // grid and ticks
    g.font = font; g.fillStyle = css("--muted"); g.strokeStyle = css("--line");
    g.lineWidth = 1;
    const xs = niceStep(x1 - x0, small ? 4 : 7), ys = niceStep(y1 - y0, small ? 4 : 6);
    g.setLineDash([2, 3]);
    g.textAlign = "center"; g.textBaseline = "top";
    for (let v = Math.ceil(x0 / xs) * xs; v <= x1 + 1e-9 * xs; v += xs) {
      g.beginPath(); g.moveTo(X(v), m.t); g.lineTo(X(v), m.t + ph); g.stroke();
      g.fillText(fmt(v), X(v), m.t + ph + 4);
    }
    g.textAlign = "right"; g.textBaseline = "middle";
    for (let v = Math.ceil(y0 / ys) * ys; v <= y1 + 1e-9 * ys; v += ys) {
      g.beginPath(); g.moveTo(m.l, Y(v)); g.lineTo(m.l + pw, Y(v)); g.stroke();
      g.fillText(fmt(v), m.l - 4, Y(v));
    }
    g.setLineDash([]);
    g.strokeRect(m.l, m.t, pw, ph);
    g.textAlign = "right"; g.textBaseline = "bottom";
    if (panel.xlabel) g.fillText(panel.xlabel, m.l + pw, h - 1);
    g.textAlign = "left"; g.textBaseline = "top";
    if (panel.ylabel) g.fillText(panel.ylabel, m.l + 4, m.t + 3);

    // series, clipped to the plot area
    g.save();
    g.beginPath(); g.rect(m.l, m.t, pw, ph); g.clip();
    const colour = { trace: "--accent", line: "--accent", mask: "--fail",
                     ref: "--muted", points: "--accent" };
    for (const s of panel.series) {
      const style = s.style || "line";
      g.strokeStyle = g.fillStyle = css(colour[style] || "--accent");
      if (style === "points") {
        g.globalAlpha = 0.55;
        const r = small ? 1.6 : 1.8;
        for (let i = 0; i < s.x.length; i++) {
          if (s.x[i] === null || s.y[i] === null) continue;
          g.fillRect(X(s.x[i]) - r / 2, Y(s.y[i]) - r / 2, r, r);
        }
        g.globalAlpha = 1;
        continue;
      }
      g.lineWidth = style === "mask" ? 1.2 : 1;
      g.setLineDash(style === "mask" ? [5, 3] : style === "ref" ? [2, 3] : []);
      g.beginPath();
      let pen = false;
      for (let i = 0; i < s.x.length; i++) {
        if (s.x[i] === null || s.y[i] === null) { pen = false; continue; }
        const px = X(s.x[i]), py = Y(s.y[i]);
        if (pen) g.lineTo(px, py); else { g.moveTo(px, py); pen = true; }
      }
      g.stroke();
      g.setLineDash([]);
    }
    if (opts.cursor) {                      // crosshair from the readout
      g.strokeStyle = css("--ink"); g.globalAlpha = 0.5;
      g.beginPath();
      g.moveTo(X(opts.cursor.x), m.t); g.lineTo(X(opts.cursor.x), m.t + ph);
      g.moveTo(m.l, Y(opts.cursor.y)); g.lineTo(m.l + pw, Y(opts.cursor.y));
      g.stroke(); g.globalAlpha = 1;
    }
    g.restore();
    // what the readout needs to map a pointer back to data
    canvas._map = { m, pw, ph, x0, x1, y0, y1 };
  }

  /* Data under the pointer: on a curve, the nearest sample of the first
   * curve (so the value read is a value the report computed); on a
   * scatter, the pointer position itself. */
  function pick(canvas, panel, clientX, clientY) {
    const map = canvas._map;
    if (!map || !panel) return null;
    const r = canvas.getBoundingClientRect();
    const px = clientX - r.left, py = clientY - r.top;
    const x = map.x0 + (px - map.m.l) / map.pw * (map.x1 - map.x0);
    const y = map.y1 - (py - map.m.t) / map.ph * (map.y1 - map.y0);
    if (x < map.x0 || x > map.x1 || y < map.y0 || y > map.y1) return null;
    const curve = panel.series.find(s => s.style !== "points" && s.style !== "mask"
                                         && s.style !== "ref");
    // a trajectory is not a function of x: read the pointer there
    if (!curve || panel.id === "iq") return { x, y };
    let lo = 0, hi = curve.x.length - 1;     // x is monotonic on these
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (curve.x[mid] < x) lo = mid; else hi = mid;
    }
    const i = Math.abs(curve.x[hi] - x) < Math.abs(curve.x[lo] - x) ? hi : lo;
    return { x: curve.x[i], y: curve.y[i] };
  }

  window.Charts = { draw, pick, fmt };
})();
