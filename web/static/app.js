"use strict";

const $ = (sel) => document.querySelector(sel);

// -- helpers -----------------------------------------------------------------

function toast(message, kind) {
  const el = $("#toast");
  el.textContent = message;
  el.className = "toast show" + (kind ? " " + kind : "");
  clearTimeout(el._timer);
  el._timer = setTimeout(() => (el.className = "toast"), 3500);
}

async function api(path, options) {
  const resp = await fetch(path, options);
  let data = {};
  try { data = await resp.json(); } catch (_) { /* empty */ }
  if (!resp.ok) {
    throw new Error(data && data.error ? data.error : "HTTP " + resp.status);
  }
  return data;
}

const fmtNum = (v, digits) => {
  if (v === null || v === undefined || v === "") return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString("en-US", { minimumFractionDigits: digits || 2, maximumFractionDigits: digits || 2 });
};

const pct = (v) => {
  if (v === null || v === undefined || v === "") return "—";
  return fmtNum(Number(v) * 100, 1) + "%";
};

// Fixed Vietnam time (UTC+7, no DST) rendering for ISO-8601 UTC timestamps.
const fmtVn = (iso) => {
  if (!iso) return "—";
  const s = String(iso).includes("Z") || String(iso).includes("+") ? String(iso) : String(iso) + "Z";
  const ms = Date.parse(s);
  if (!Number.isFinite(ms)) return String(iso);
  const e = new Date(ms + 7 * 3600 * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(e.getUTCDate())}/${p(e.getUTCMonth() + 1)}/${e.getUTCFullYear()} ${p(e.getUTCHours())}:${p(e.getUTCMinutes())}:${p(e.getUTCSeconds())} (+07)`;
};

// Compact ``DD/MM HH:MM`` label for a UTC epoch-millisecond candle timestamp in Vietnam time.
const fmtVnMs = (ms) => {
  if (!Number.isFinite(Number(ms))) return "—";
  const e = new Date(Number(ms) + 7 * 3600 * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(e.getUTCDate())}/${p(e.getUTCMonth() + 1)} ${p(e.getUTCHours())}:${p(e.getUTCMinutes())}`;
};

function badge(text) {
  const cls = String(text).toLowerCase().replace("_", "-");
  return `<span class="badge ${cls}">${text}</span>`;
}

function emptyBox(text) {
  return `<p class="muted">${text}</p>`;
}

// -- navigation ---------------------------------------------------------------

const PANELS = ["dashboard", "signals", "trades", "statistics", "settings"];
let current = "dashboard";

document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    current = btn.dataset.panel;
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + current));
    const titles = { dashboard: ["Dashboard", "Signal & demo trading overview"], signals: ["Signals", "AI signal ledger"], trades: ["Trades", "Demo trade ledger"], statistics: ["Statistics", "Performance metrics"], settings: ["Settings", "AI provider, demo account & notifications"] };
    $("#pageTitle").textContent = titles[current][0];
    $("#pageSub").textContent = titles[current][1];
    refresh(current);
  });
});

// -- dashboard ----------------------------------------------------------------

function renderHealth(health) {
  const pill = (state, ok) => `<span class="status-pill ${ok ? "ok" : "warn"}">${state}</span>`;
  const rows = `
    <div class="kv"><span>state</span><span>${pill(health.state, health.state === "RUNNING")}</span></div>
    <div class="kv"><span>scheduler</span><span>${pill(health.scheduler, health.scheduler === "RUNNING")}</span></div>
    <div class="kv"><span>monitor</span><span>${pill(health.monitor, health.monitor === "RUNNING")}</span></div>
    ${health.scheduler_last_tick ? `<div class="kv"><span>last tick</span><span>${health.scheduler_last_tick}</span></div>` : ""}
    ${health.monitor_last_poll ? `<div class="kv"><span>last poll</span><span>${health.monitor_last_poll}</span></div>` : ""}
    ${health.last_error ? `<div class="kv"><span>last error</span><span>${escapeHtml(health.last_error)}${health.last_error_at ? ` <span class="hint" style="color:var(--muted)">(${fmtVn(health.last_error_at)})</span>` : ""}</span></div>` : ""}`;
  return `<div class="form" style="gap:2px">${rows}</div>`;
}

function renderSignalPanel(signal, position, positionOutcomes, unrealized) {
  if (!signal) return emptyBox("No signal has been created yet. The daemon analyzes " + (window._symbol || "BTCUSDT") + " on each closed 1H candle.");
  const dir = signal.direction;
  const rows = `
    <div class="kv"><span>Direction</span><span>${badge(dir)}</span></div>
    <div class="kv"><span>Entry</span><span>$${fmtNum(signal.entry, 2)}</span></div>
    <div class="kv"><span>Take Profit</span><span>$${fmtNum(signal.take_profit, 2)}</span></div>
    <div class="kv"><span>Stop Loss</span><span>$${fmtNum(signal.stop_loss, 2)}</span></div>
    <div class="kv"><span>Confidence</span><span>${fmtNum(signal.confidence, 0)}/100</span></div>
    <div class="kv"><span>Model</span><span>${signal.provider} / ${signal.model_name}</span></div>
    <div class="kv"><span>Analysis</span><span>${fmtVn(signal.analysis_timestamp)}</span></div>`;
  const positionRows = position ? `
    <div class="kv"><span>Side</span><span>${badge(position.side)}</span></div>
    <div class="kv"><span>Quantity</span><span>${fmtNum(position.quantity, 6)} BTC</span></div>
    <div class="kv"><span>Margin / Size</span><span>${fmtNum(position.margin, 2)} / $${fmtNum(position.position_size, 2)}</span></div>
    <div class="kv"><span>Leverage</span><span>${position.leverage}x</span></div>` : "";
  const unrealRow = unrealized ? `
    <div class="kv"><span>Open PnL</span><span style="color:${Number(unrealized.gross_pnl) >= 0 ? "var(--green)" : "var(--red)"};font-weight:600">${Number(unrealized.gross_pnl) >= 0 ? "+" : ""}$${fmtNum(unrealized.gross_pnl, 2)} (${fmtNum(unrealized.pnl_percent, 2)}% margin)</span></div>` : "";
  const outcomeRows = positionOutcomes ? `
    <div class="kv"><span>TP outcome</span><span>@$${fmtNum(positionOutcomes.take_profit.exit_price, 2)} → ${fmtNum(positionOutcomes.take_profit.net_pnl, 2)} (${fmtNum(positionOutcomes.take_profit.pnl_percent, 2)}%) · bal. $${fmtNum(positionOutcomes.take_profit.projected_balance, 2)}</span></div>
    <div class="kv"><span>SL outcome</span><span>@$${fmtNum(positionOutcomes.stop_loss.exit_price, 2)} → ${fmtNum(positionOutcomes.stop_loss.net_pnl, 2)} (${fmtNum(positionOutcomes.stop_loss.pnl_percent, 2)}%) · bal. $${fmtNum(positionOutcomes.stop_loss.projected_balance, 2)}</span></div>` : "";
  return `<div class="form" style="gap:2px">${rows}${positionRows}${unrealRow}${outcomeRows}</div>`;
}

async function loadDashboard() {
  const d = await api("/api/dashboard");
  const acc = d.account;
  const stats = d.statistics;
  const health = d.health;

  let cards = `
    <div class="card metric"><div class="metric-label">Balance</div><div class="metric-value">$${acc ? fmtNum(acc.balance, 2) : "—"}</div></div>
    <div class="card metric"><div class="metric-label">Equity</div><div class="metric-value">$${acc ? fmtNum(acc.equity, 2) : "—"}</div></div>
    <div class="card metric"><div class="metric-label">Net PnL</div><div class="metric-value">$${stats ? fmtNum(stats.net_pnl, 2) : "—"}</div></div>
    <div class="card metric"><div class="metric-label">Win Rate</div><div class="metric-value">${stats ? pct(stats.win_rate) : "—"}</div></div>
    <div class="card metric"><div class="metric-label">Peak Equity</div><div class="metric-value">$${acc ? fmtNum(acc.peak_equity, 2) : "—"}</div></div>
    <div class="card metric"><div class="metric-label">Max Drawdown</div><div class="metric-value">${stats ? pct(stats.max_drawdown) : "—"}</div></div>`;
  $("#dashboardCards").innerHTML = cards;
  $("#currentSignal").innerHTML = renderSignalPanel(d.active_signal, d.position, d.position_outcomes, d.unrealized);
  $("#systemHealth").innerHTML = renderHealth(health);

  const statusEl = $("#statusPill");
  window._activeSignal = d.active_signal;
  window._symbol = d.symbol || window._symbol || "BTCUSDT";
  $("#sideSymbol").textContent = window._symbol;
  $("#priceCardTitle").textContent = window._symbol + " Price";
  if (d.active_signal) {
    statusEl.className = "status-pill pending";
    statusEl.textContent = "SIGNAL ACTIVE — " + d.active_signal.status.toUpperCase();
  } else {
    statusEl.className = "status-pill ok";
    statusEl.textContent = "IDLE — READY";
  }
}

// -- price chart ----------------------------------------------------------------

function tsMillis(t) {
  return t > 9999999999 ? t : t * 1000;
}

function drawPriceChart(canvas, chart, view) {
  const parent = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const width = parent.clientWidth;
  const height = canvas.clientHeight || 360;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const allCandles = chart.candles || [];
  const overlay = chart.overlay || {};
  const total = allCandles.length;
  if (!total) { $("#chartMsg").textContent = "No candle data yet."; return; }
  $("#chartMsg").textContent = "";

  const end = total - (view ? view.back : 0);
  const start = Math.max(0, end - (view ? view.count : total));
  const candles = allCandles.slice(start, end);
  const n = candles.length;

  const pad = { top: 8, right: 60, bottom: 18, left: 8 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const volH = plotH * 0.2;
  const priceH = plotH - volH;

  let min = Infinity, max = -Infinity;
  for (const c of candles) { if (c.l < min) min = c.l; if (c.h > max) max = c.h; }
  if (overlay.has_signal) {
    for (const p of [overlay.entry, overlay.take_profit, overlay.stop_loss]) {
      if (p < min) min = p;
      if (p > max) max = p;
    }
  }
  let spread = max - min;
  if (!spread || spread <= 0) spread = max || 1;
  const padRange = spread * 0.05;
  min -= padRange; max += padRange;
  const yFor = (price) => pad.top + priceH - ((price - min) / (max - min)) * priceH;

  const slotW = plotW / n;
  const bodyW = Math.max(1, slotW * 0.62);
  const font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";

  ctx.font = font;
  const step = (max - min) / 4;
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const price = min + step * i;
    const y = yFor(price);
    ctx.strokeStyle = "rgba(128,128,128,0.13)";
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
    ctx.fillStyle = "rgba(160,160,160,0.85)";
    ctx.fillText(price.toFixed(0), width - 8, y);
  }

  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const xEvery = Math.max(1, Math.ceil(n / 6));
  for (let i = 0; i < n; i += xEvery) {
    ctx.fillStyle = "rgba(160,160,160,0.85)";
    ctx.fillText(fmtVnMs(tsMillis(candles[i].t)), pad.left + slotW * i + slotW / 2, pad.top + priceH + 4);
  }

  for (let i = 0; i < n; i++) {
    const c = candles[i];
    const x = pad.left + slotW * i + slotW / 2;
    const color = c.c >= c.o ? "#26a69a" : "#ef5350";
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, yFor(c.h)); ctx.lineTo(x, yFor(c.l)); ctx.stroke();
    const yTop = Math.min(yFor(c.o), yFor(c.c));
    const bodyH = Math.max(1, Math.abs(yFor(c.o) - yFor(c.c)));
    ctx.fillRect(x - bodyW / 2, yTop, bodyW, bodyH);
  }

  let maxVol = 0;
  for (const c of candles) if (c.v > maxVol) maxVol = c.v;
  if (maxVol > 0) {
    for (let i = 0; i < n; i++) {
      const c = candles[i];
      const x = pad.left + slotW * i + slotW / 2;
      const vh = (c.v / maxVol) * volH;
      ctx.fillStyle = c.c >= c.o ? "rgba(38,166,154,0.35)" : "rgba(239,83,80,0.35)";
      ctx.fillRect(x - bodyW / 2, pad.top + priceH + volH - vh, bodyW, vh);
    }
  }

  const chipH = 16;
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  if (overlay.has_signal) {
    const levels = [
      { price: Number(overlay.stop_loss), text: "SL " + fmtNum(overlay.stop_loss, 1), color: "#ef5350" },
      { price: Number(overlay.entry), text: "Entry " + fmtNum(overlay.entry, 1), color: "#42a5f5" },
      { price: Number(overlay.take_profit), text: "TP " + fmtNum(overlay.take_profit, 1), color: "#26a69a" },
    ];
    for (const l of levels) {
      const y = yFor(l.price);
      if (y < pad.top - 1 || y > pad.top + priceH + 1) continue;
      ctx.strokeStyle = l.color; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
      ctx.setLineDash([]);
      const tw = ctx.measureText(l.text).width + 10;
      ctx.fillStyle = l.color;
      ctx.fillRect(pad.left + 2, y - 8, tw, chipH);
      ctx.fillStyle = "#0f1115"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText(l.text, pad.left + 2 + 5, y);
    }
  }

  const newest = candles[n - 1];
  if (newest && end === total) {
    const yLast = yFor(newest.c);
    if (yLast >= pad.top && yLast <= pad.top + priceH) {
      const rightEdge = pad.left + slotW * n;
      const text = fmtNum(newest.c, 1);
      const tw = ctx.measureText(text).width + 10;
      const chipX = Math.max(rightEdge + 4, width - tw - 4);
      const chipW = Math.min(tw, width - chipX - 4);
      ctx.strokeStyle = "rgba(245,197,66,0.55)"; ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.left, yLast); ctx.lineTo(Math.max(pad.left, chipX - 4), yLast); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#f5c542";
      ctx.fillRect(chipX, yLast - 8, chipW, chipH);
      ctx.fillStyle = "#0f1115"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText(text, chipX + 5, yLast);
    }
  }
}

let chartView = null;
let lastChart = null;
const CHART_WINDOW_DEFAULT = 160;
const CHART_WINDOW_MIN = 15;

function clampView(n) {
  const stored = chartView || { count: Math.min(CHART_WINDOW_DEFAULT, n), backFromNewest: 0 };
  const count = Math.max(CHART_WINDOW_MIN, Math.min(stored.count, n));
  const back = Math.max(0, Math.min(stored.backFromNewest, n - count));
  return { count, back };
}

function updateChartRange(view, candles) {
  const el = $("#chartRange");
  if (!el) return;
  const n = candles.length;
  const endC = n - view.back;
  const start = Math.max(0, endC - view.count);
  if (start === 0 && endC === n) {
    el.textContent = "all · " + n + " candles";
    return;
  }
  const fmtT = (c) => fmtVnMs(tsMillis(c.t));
  el.textContent = fmtT(candles[start]) + " → " + fmtT(candles[endC - 1]) + " · " + (endC - start) + " candles";
}

function redrawChart() {
  if (!lastChart || !lastChart.candles || !lastChart.candles.length) return;
  const candles = lastChart.candles;
  const view = clampView(candles.length);
  drawPriceChart($("#priceChart"), lastChart, view);
  updateChartRange(view, candles);
}

function zoomChart(factor) {
  if (!lastChart || !lastChart.candles.length) return;
  const n = lastChart.candles.length;
  const cur = clampView(n);
  chartView = {
    count: Math.max(CHART_WINDOW_MIN, Math.min(Math.round(cur.count * factor), n)),
    backFromNewest: 0,
  };
  redrawChart();
}

function panChart(direction) {
  if (!lastChart || !lastChart.candles.length) return;
  const n = lastChart.candles.length;
  const cur = clampView(n);
  const step = Math.max(1, Math.round(cur.count * 0.2));
  chartView = {
    count: cur.count,
    backFromNewest: Math.max(0, Math.min(cur.back + direction * step, n - cur.count)),
  };
  redrawChart();
}

function resetChart() {
  chartView = null;
  redrawChart();
}

const chartPad = { left: 62, right: 8 };
let chartDrag = null;

$("#priceChart").addEventListener("mousedown", (e) => {
  if (!lastChart || !lastChart.candles.length) return;
  const cur = clampView(lastChart.candles.length);
  chartDrag = { x: e.clientX, back: cur.back, count: cur.count };
  e.currentTarget.classList.add("dragging");
  e.preventDefault();
});
window.addEventListener("mousemove", (e) => {
  if (!chartDrag) return;
  const canvas = $("#priceChart");
  const n = lastChart.candles.length;
  const slotW = (canvas.clientWidth - chartPad.left - chartPad.right) / chartDrag.count;
  const delta = Math.round((e.clientX - chartDrag.x) / slotW);
  chartView = {
    count: chartDrag.count,
    backFromNewest: Math.max(0, Math.min(chartDrag.back + delta, n - chartDrag.count)),
  };
  redrawChart();
});
window.addEventListener("mouseup", () => {
  if (!chartDrag) return;
  chartDrag = null;
  $("#priceChart").classList.remove("dragging");
});

$("#priceChart").addEventListener("wheel", (e) => {
  e.preventDefault();
  zoomChart(e.deltaY < 0 ? 1 / 1.25 : 1.25);
}, { passive: false });

$("#chartZoomIn").addEventListener("click", () => zoomChart(1 / 1.25));
$("#chartZoomOut").addEventListener("click", () => zoomChart(1.25));
$("#chartPanLeft").addEventListener("click", () => panChart(1));
$("#chartPanRight").addEventListener("click", () => panChart(-1));
$("#chartReset").addEventListener("click", resetChart);

async function loadChart() {
  const interval = $("#chartInterval").value;
  try {
    const data = (await api("/api/chart?interval=" + interval + "&limit=500")).chart;
    lastChart = data;
    redrawChart();
    $("#chartLive").textContent = "updated " + new Date().toTimeString().slice(0, 8);
    const o = data.overlay || {};
    const candles = data.candles || [];
    let parts = [window._symbol || "BTCUSDT", interval.toUpperCase()];
    if (o.has_signal) {
      parts.push(o.direction + (o.position_open ? " · position OPEN" : " · " + o.status));
    }
    if (candles.length) parts.push("last $" + fmtNum(candles[candles.length - 1].c, 2));
    $("#chartLegend").innerHTML = '<span class="muted">' + parts.join("  ·  ") + '</span>';
  } catch (err) {
    $("#chartMsg").textContent = "Chart data unavailable — " + err.message;
    lastChart = null;
    $("#chartLive").textContent = "";
  }
}

// -- signal chat (read-only Q&A drawer) ----------------------------------------

function openSignalChat() {
  const active = window._activeSignal;
  $("#chatContext").textContent = active
    ? ("#" + active.id + " · " + active.direction + " · " + fmtVn(active.created_at))
    : "no active signal — chat will explain once one exists";
  $("#chatDrawer").classList.add("open");
  $("#chatDrawer").setAttribute("aria-hidden", "false");
  $("#chatBackdrop").classList.remove("hidden");
  $("#chatInput").focus();
}

function closeSignalChat() {
  $("#chatDrawer").classList.remove("open");
  $("#chatDrawer").setAttribute("aria-hidden", "true");
  $("#chatBackdrop").classList.add("hidden");
}

function chatMsg(kind, text) {
  const el = document.createElement("div");
  el.className = "msg " + kind;
  el.textContent = text;
  $("#chatMessages").appendChild(el);
  $("#chatMessages").scrollTop = $("#chatMessages").scrollHeight;
  return el;
}

async function sendSignalChat(questionText) {
  const question = (questionText || $("#chatInput").value).trim();
  if (!question) return;
  $("#chatInput").value = "";
  chatMsg("user", question);
  const bot = chatMsg("bot", "…");
  bot.classList.add("typing");
  const sendBtn = $("#chatSend");
  sendBtn.disabled = true;
  let first = true;
  try {
    const resp = await fetch("/api/signal-chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question }),
    });
    if (!resp.ok) {
      let message = "HTTP " + resp.status;
      try { message = (await resp.json()).error || message; } catch (_) { /* empty */ }
      throw new Error(message);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line) continue;
        let event;
        try { event = JSON.parse(line); } catch (_) { continue; }
        if (event.delta) {
          if (first) { bot.textContent = ""; bot.classList.remove("typing"); first = false; }
          bot.textContent += event.delta;
        } else if (event.text) {
          if (first) { bot.classList.remove("typing"); first = false; }
          bot.textContent = event.text;
        } else if (event.error) {
          if (first) { bot.classList.remove("typing"); first = false; }
          bot.textContent = event.error;
          bot.classList.add("error");
        }
      }
      $("#chatMessages").scrollTop = $("#chatMessages").scrollHeight;
    }
    if (first) { bot.classList.remove("typing"); bot.textContent = "—"; }
  } catch (err) {
    bot.classList.remove("typing");
    bot.textContent = "❌ " + err.message;
    bot.classList.add("error");
  } finally {
    sendBtn.disabled = false;
    $("#chatMessages").scrollTop = $("#chatMessages").scrollHeight;
  }
}

$("#chatOpenBtn").addEventListener("click", openSignalChat);
$("#chatClose").addEventListener("click", closeSignalChat);
$("#chatBackdrop").addEventListener("click", closeSignalChat);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSignalChat(); });
$("#chatSend").addEventListener("click", () => sendSignalChat());
$("#chatInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendSignalChat();
});
$("#chatClear").addEventListener("click", () => {
  $("#chatMessages").innerHTML = "";
  const empty = document.createElement("p");
  empty.className = "muted";
  empty.textContent = "Start a conversation about the current signal.";
  $("#chatMessages").appendChild(empty);
});
document.querySelectorAll(".chat-chip").forEach((chip) =>
  chip.addEventListener("click", () => sendSignalChat(chip.dataset.q))
);

// -- signals ------------------------------------------------------------------

const VN_OFFSET_MIN = 7 * 3600 * 1000;

function vnDayStartUtc(daysAgo) {
  const nowVn = Date.now() + VN_OFFSET_MIN;
  const startVn = Math.floor(nowVn / 86400000) * 86400000 - daysAgo * 86400000;
  return new Date(startVn - VN_OFFSET_MIN).toISOString();
}

function sigTimeRange() {
  const v = $("#sigTime").value;
  if (!v) return null;
  const days = v === "today" ? 0 : v === "7d" ? 6 : v === "30d" ? 29 : NaN;
  if (!Number.isFinite(days)) return null;
  return { since: vnDayStartUtc(days), until: new Date().toISOString() };
}

function sigSelectedCount() {
  return document.querySelectorAll("#signalsTable .sig-check:checked").length;
}

function updateSigDeleteCount() {
  const n = sigSelectedCount();
  $("#sigDeleteSel").disabled = n === 0;
  $("#sigDeleteSel").textContent = "Delete selected (" + n + ")";
}

function renderSignalTable(list) {
  if (!list.length) return emptyBox("No signals recorded yet.");
  const head = `<tr><th></th><th>ID</th><th>Direction</th><th>Entry</th><th>TP</th><th>SL</th><th>Conf</th><th>Status</th><th>Result</th><th>Model</th><th>Time</th></tr>`;
  const body = list.map((s) => `<tr>
    <td><input type="checkbox" class="sig-check" data-id="${s.id}"></td>
    <td>${s.id}</td>
    <td>${badge(s.direction)}</td>
    <td>$${fmtNum(s.entry, 2)}</td>
    <td>$${fmtNum(s.take_profit, 2)}</td>
    <td>$${fmtNum(s.stop_loss, 2)}</td>
    <td>${fmtNum(s.confidence, 0)}</td>
    <td>${badge(s.status)}</td>
    <td>${s.result ? badge(s.result) : "—"}</td>
    <td>${s.model_name}</td>
    <td>${fmtVn(s.created_at)}</td>
  </tr>`).join("");
  return `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

async function loadSignals() {
  const p = new URLSearchParams({ limit: "500" });
  const dir = $("#sigDir").value;
  const st = $("#sigStatus").value;
  if (dir) p.set("direction", dir);
  if (st) p.set("status", st);
  const range = sigTimeRange();
  if (range) {
    p.set("since", range.since);
    p.set("until", range.until);
  }
  const data = await api("/api/signals?" + p.toString());
  $("#signalsTable").innerHTML = renderSignalTable(data.signals || []);
  $("#sigSelAll").textContent = "Select all";
  updateSigDeleteCount();
}

async function deleteSignals(body) {
  const res = await api("/api/signals", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const c = res.counts || {};
  const skipped = (c.skipped_open || []).length;
  toast(
    "Deleted " + (c.deleted || []).length + " signal(s)" + (skipped ? ", kept " + skipped + " OPEN" : ""),
    skipped ? "warn" : "ok"
  );
  await Promise.allSettled([loadSignals(), loadTrades(), loadStatistics()]);
}

async function loadTrades() {
  const data = await api("/api/trades?limit=100");
  const list = data.trades || [];
  if (!list.length) { $("#tradesTable").innerHTML = emptyBox("No demo trades closed yet."); return; }
  const head = `<tr><th>ID</th><th>Side</th><th>Entry</th><th>Exit</th><th>Qty</th><th>Gross</th><th>Fee</th><th>Net PnL</th><th>Result</th><th>Closed</th></tr>`;
  const body = list.map((t) => `<tr>
    <td>${t.id}</td>
    <td>${badge(t.side)}</td>
    <td>$${fmtNum(t.entry_price, 2)}</td>
    <td>$${fmtNum(t.exit_price, 2)}</td>
    <td>${fmtNum(t.quantity, 6)}</td>
    <td>$${fmtNum(t.gross_pnl, 2)}</td>
    <td>$${fmtNum(t.fee, 4)}</td>
    <td style="color:${Number(t.net_pnl) >= 0 ? "var(--green)" : "var(--red)"}">$${fmtNum(t.net_pnl, 2)}</td>
    <td>${badge(t.result)}</td>
    <td>${fmtVn(t.closed_at)}</td>
  </tr>`).join("");
  $("#tradesTable").innerHTML = `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

async function loadStatistics() {
  const d = (await api("/api/statistics")).statistics || {};
  const cards = `
    <div class="card metric"><div class="metric-label">Net PnL</div><div class="metric-value">$${fmtNum(d.net_pnl, 2)}</div></div>
    <div class="card metric"><div class="metric-label">Win Rate</div><div class="metric-value">${pct(d.win_rate)}</div></div>
    <div class="card metric"><div class="metric-label">Trades</div><div class="metric-value">${fmtNum(d.total_trades, 0)}</div></div>
    <div class="card metric"><div class="metric-label">Wins / Losses</div><div class="metric-value">${fmtNum(d.wins, 0)} / ${fmtNum(d.losses, 0)}</div></div>
    <div class="card metric"><div class="metric-label">Max Drawdown</div><div class="metric-value">${pct(d.max_drawdown)}</div></div>
    <div class="card metric"><div class="metric-label">Total Fees</div><div class="metric-value">$${fmtNum(d.total_fees, 4)}</div></div>`;
  $("#statsCards").innerHTML = cards;
  const detail = Object.entries(d).filter(([k]) => !["wins", "losses", "total_trades"].includes(k))
    .map(([k, v]) => `<div class="kv"><span>${k.replace(/_/g, " ")}</span><span>${fmtNum(v, 4)}</span></div>`).join("");
  $("#statsDetail").innerHTML = `<div class="form" style="gap:2px">${detail}</div>`;
}

// -- settings -----------------------------------------------------------------

const PRESETS = ["openai", "deepseek", "openai_compatible", "custom"];
let settingsCache = null;

async function loadSettings() {
  const s = (await api("/api/settings")).settings;
  settingsCache = s;
  renderSettings(s);
}

function renderSettings(s) {
  const ai = s.ai_provider;
  const state = s.config_state;
  const stateBadge = state === "pending"
    ? `<span class="status-pill pending">CHANGE WILL APPLY WHEN IDLE</span>`
    : `<span class="status-pill ok">APPLIED</span>`;

  const provider = ai.provider;
  const radio = `<div class="radio-row">
    <label><input type="radio" name="provider" value="ollama" ${provider === "ollama" ? "checked" : ""}> Ollama (local)</label>
    <label><input type="radio" name="provider" value="api" ${provider === "api" ? "checked" : ""}> API (OpenAI-compatible)</label>
  </div>`;

  const presetRow = `<div class="field" id="presetField" style="${provider === "api" ? "" : "display:none"}">
    <label>API Preset</label>
    <select id="preset">${PRESETS.map((p) => `<option value="${p}" ${ai.preset === p ? "selected" : ""}>${p}</option>`).join("")}</select>
  </div>`;

  const fields = `
    ${radio}
    ${presetRow}
    <div class="field"><label>Base URL</label><input type="text" id="baseUrl" value="${escapeHtml(ai.base_url || "")}" placeholder="${provider === "ollama" ? "e.g. http://localhost:11434" : "e.g. https://api.openai.com/v1"}" /></div>
    <div class="field"><label>Model</label><input type="text" id="model" value="${escapeHtml(ai.model || "")}" placeholder="qwen3:4b" /></div>
    <div class="field"><label>Analysis Mode</label><select id="analysisMode">
      <option value="single" ${ai.analysis_mode === "multi" ? "" : "selected"}>Single (1 AI call)</option>
      <option value="multi" ${ai.analysis_mode === "multi" ? "selected" : ""}>Multi-agent debate (5 AI calls)</option>
    </select></div>
    <div class="hint">"Multi" runs the analyst &gt; bull &gt; bear &gt; trader &gt; risk-manager committee (5 LLM calls per closed 1H candle, same provider/model). Applies to the NEXT analysis once no signal is active.</div>
    <div class="row2">
      <div class="field"><label>Temperature (0–5)</label><input type="number" step="0.1" min="0" max="5" id="temperature" value="${ai.temperature}" /></div>
      <div class="field"><label>Max Tokens</label><input type="number" step="1" min="1" id="maxTokens" value="${ai.max_tokens}" /></div>
    </div>
    <div class="row2">
      <div class="field"><label>Timeout (s)</label><input type="number" step="1" min="1" id="timeout" value="${ai.timeout}" /></div>
      <div class="field"><label>API Key (API preset)</label><input type="password" id="apiKey" placeholder="${ai.api_key_configured ? "configured: " + (ai.api_key_masked || "••••") : "optional"}" autocomplete="off" /></div>
    </div>
    <div class="hint">AI provider changes apply to the NEXT analysis. ${state === "pending" ? "A change is waiting — it will apply automatically when no signal is active." : ""}</div>
    <div class="row2">
      <button type="submit" class="btn">Save AI Provider</button>
      <button type="button" class="btn secondary" id="testConn">Test Connection</button>
    </div>
    <div id="connResult">${stateBadge}</div>`;

  $("#aiForm").innerHTML = fields;

  $("#aiForm").querySelectorAll('input[name="provider"]').forEach((r) =>
    r.addEventListener("change", () => {
      const isApi = $("#aiForm").querySelector('input[name="provider"]:checked').value === "api";
      $("#presetField").style.display = isApi ? "" : "none";
    })
  );

  $("#aiForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    await saveAiProvider();
  });
  $("#testConn").addEventListener("click", testConnection);

  renderDemoForm(s.demo);
  renderTelegramForm(s.telegram);
  renderSymbolForm(s.symbol);
  renderRuntime(s.runtime);
  renderAudit(s.audit);
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function aiPayload() {
  const f = $("#aiForm");
  const provider = f.querySelector('input[name="provider"]:checked').value;
  const apiKey = $("#apiKey").value.trim();
  return {
    provider,
    preset: provider === "api" ? $("#preset").value : "ollama",
    base_url: $("#baseUrl").value.trim(),
    model: $("#model").value.trim(),
    temperature: Number($("#temperature").value),
    max_tokens: Number($("#maxTokens").value),
    timeout: Number($("#timeout").value),
    api_key: apiKey,
    analysis_mode: $("#analysisMode").value,
  };
}

async function saveAiProvider() {
  try {
    const payload = aiPayload();
    const result = await api("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ai_provider: payload }) });
    toast("AI provider saved — applies to next analysis.");
    renderSettings(result.settings);
    refresh("settings");
  } catch (err) {
    toast("AI provider save failed: " + err.message, "error");
  }
}

async function testConnection() {
  const payload = aiPayload();
  const btn = $("#testConn");
  btn.disabled = true;
  btn.textContent = "Testing…";
  try {
    const result = (await api("/api/settings/test-connection", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })).result;
    const ms = result.latency_ms;
    const status = result.success
      ? `<span class="status-pill ok">OK · ${ms}ms · ${result.model_available === false ? "model NOT found on server" : "reachable"}</span>`
      : `<span class="status-pill bad">FAILED (${ms}ms)</span>`;
    const models = result.available_models && result.available_models.length
      ? `<div class="hint">Models: ${result.available_models.join(", ")}</div>` : "";
    $("#connResult").innerHTML = `${status}${models}${result.error ? `<div class="hint" style="color:var(--red)">${escapeHtml(result.error)}</div>` : ""}`;
  } catch (err) {
    $("#connResult").innerHTML = `<span class="status-pill bad">FAILED</span><div class="hint" style="color:var(--red)">${escapeHtml(err.message)}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Test Connection";
  }
}

function renderDemoForm(demo) {
  const lockNote = demo.locked
    ? `<div class="lock-note">Demo settings are LOCKED while a signal is active (AI Lock + one-active-signal rule).</div>` : "";
  const fields = `
    ${lockNote}
    <div class="field"><label>Initial Balance (USDT)</label><input type="number" step="0.01" id="dBalance" value="${demo.initial_balance}" ${demo.locked ? "disabled" : ""} /></div>
    <div class="row2">
      <div class="field"><label>Margin per Trade (USDT)</label><input type="number" step="0.01" id="dMargin" value="${demo.margin_per_trade}" ${demo.locked ? "disabled" : ""} /></div>
      <div class="field"><label>Leverage (x)</label><input type="number" step="1" min="1" id="dLeverage" value="${demo.leverage}" ${demo.locked ? "disabled" : ""} /></div>
    </div>
    <div class="row2">
      <div class="field"><label>Risk %</label><input type="number" step="0.01" id="dRisk" value="${demo.risk_percent}" ${demo.locked ? "disabled" : ""} /></div>
      <div class="field"><label>Fee Rate (fraction, e.g. 0.0004 = 0.04%)</label><input type="number" step="0.0001" id="dFee" value="${demo.fee_rate}" ${demo.locked ? "disabled" : ""} /></div>
    </div>
    <div class="hint">Changes apply to FUTURE trades only — the ledger and current position are never touched. To apply a new Initial Balance, Save it, then click Reset Demo Account.</div>
    <div class="row2">
      <button type="button" class="btn" id="saveDemo" ${demo.locked ? "disabled" : ""}>Save Demo Settings</button>
      <button type="button" class="btn danger" id="resetDemo" ${demo.locked ? "disabled" : ""}>Reset Demo Account</button>
    </div>`;
  $("#demoForm").innerHTML = fields;
  $("#saveDemo").addEventListener("click", async () => {
    try {
      const body = {
        initial_balance: $("#dBalance").value,
        margin_per_trade: $("#dMargin").value,
        leverage: Number($("#dLeverage").value),
        risk_percent: $("#dRisk").value,
        fee_rate: $("#dFee").value,
      };
      const result = await api("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ demo: body }) });
      toast("Demo settings saved for future trades.");
      renderSettings(result.settings);
    } catch (err) {
      toast("Demo save failed: " + err.message, "error");
    }
  });
  $("#resetDemo").addEventListener("click", async () => {
    const ok = window.confirm(
      "Reset the demo account to a fresh start?\n\n" +
      "Balance / Equity / Peak Equity are set to the saved Initial Balance,\n" +
      "and ALL demo positions and trades are cleared.\n" +
      "Signal history is kept."
    );
    if (!ok) return;
    try {
      const result = await api("/api/demo/reset", { method: "POST" });
      toast("Demo account reset to a fresh start.");
      renderSettings(result.settings);
      refresh("dashboard");
    } catch (err) {
      toast("Demo reset failed: " + err.message, "error");
    }
  });
}

function renderTelegramForm(t) {
  const fields = `
    <div class="radio-row">
      <label><input type="checkbox" id="tgEnabled" ${t.enabled ? "checked" : ""} /> Enabled</label>
    </div>
    <div class="field"><label>Chat ID</label><input type="text" id="tgChat" value="${escapeHtml(t.chat_id_masked === "********" ? "" : t.chat_id_masked)}" placeholder="e.g. 123456789" />${t.chat_id_configured ? `<span class="hint">configured (masked)</span>` : ""}</div>
    <div class="field"><label>Bot Token</label><input type="password" id="tgToken" autocomplete="off" placeholder="${t.token_configured ? "configured: " + (t.token_masked || "••••") : "not configured"}" /></div>
    <div class="hint">Tokens and chat ids are stored encrypted-side (0600 secrets file); the UI always masks them.</div>
    <div class="row2">
      <button type="button" class="btn" id="saveTelegram">Save Telegram</button>
      <button type="button" class="btn secondary" id="testTelegram">Send Test</button>
    </div>`;
  $("#telegramForm").innerHTML = fields;
  $("#saveTelegram").addEventListener("click", async () => {
    try {
      const body = {
        enabled: $("#tgEnabled").checked,
        chat_id: $("#tgChat").value.trim(),
        bot_token: $("#tgToken").value.trim(),
      };
      const result = await api("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ telegram: body }) });
      toast("Telegram settings saved.");
      renderSettings(result.settings);
    } catch (err) {
      toast("Telegram save failed: " + err.message, "error");
    }
  });
  $("#testTelegram").addEventListener("click", async () => {
    try {
      const r = await api("/api/settings/test-telegram", { method: "POST" });
      toast(r.result.success ? "Test message sent ✓" : "Test failed: " + r.result.error, r.result.success ? "" : "error");
    } catch (err) { toast("Test failed: " + err.message, "error"); }
  });
}

function renderSymbolForm(sym) {
  window._symbol = sym.symbol || window._symbol || "BTCUSDT";
  $("#sideSymbol").textContent = window._symbol;
  $("#priceCardTitle").textContent = window._symbol + " Price";
  const select = $("#symbolSelect");
  select.innerHTML = (sym.supported || ["BTCUSDT", "ETHUSDT", "XAUUSDT"]).map((s) => `<option value="${s}">${s}</option>`).join("");
  select.value = sym.symbol;
  $("#saveSymbol").addEventListener("click", async () => {
    try {
      const chosen = $("#symbolSelect").value;
      const result = await api("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ symbol: { symbol: chosen } }) });
      toast("Active symbol set to " + chosen + ".");
      renderSettings(result.settings);
      if (current === "dashboard") refresh("dashboard");
    } catch (err) {
      toast("Symbol save failed: " + err.message, "error");
    }
  });

  $("#clearErrorBtn").addEventListener("click", async () => {
    if (!confirm("Clear the stored \"last error\" shown in System Health?")) return;
    try {
      const result = await api("/api/runtime/clear-error", { method: "POST" });
      toast("Last error cleared.");
      const h = result.health || {};
      $("#systemHealth").innerHTML = renderHealth(h);
      if (current === "dashboard") refresh("dashboard");
    } catch (err) {
      toast("Clear failed: " + err.message, "error");
    }
  });
}

function renderRuntime(runtime) {
  const rows = `
    <div class="kv"><span>Symbol</span><span>${runtime.symbol}</span></div>
    <div class="kv"><span>Timeframe (analysis)</span><span>${runtime.timeframe}</span></div>
    <div class="kv"><span>Scheduler poll</span><span>${runtime.scheduler_poll}s</span></div>
    <div class="kv"><span>Monitor poll</span><span>${runtime.monitor_poll}s</span></div>
    <div class="kv"><span>Database</span><span class="${runtime.db_connected ? "" : "status-pill bad"}">${escapeHtml(runtime.db_path)}</span></div>`;
  $("#runtimeBox").innerHTML = `<div class="form" style="gap:2px">${rows}</div>`;
}

function renderAudit(audit) {
  const list = audit || [];
  if (!list.length) {
    $("#auditBox").innerHTML = emptyBox("No audit entries yet.");
    return;
  }
  const head = `<tr><th>Time</th><th>Namespace</th><th>Action</th><th>Summary</th></tr>`;
  const body = list.map((a) => `<tr>
    <td>${fmtVn(a.created_at)}</td>
    <td>${escapeHtml(a.namespace)}</td>
    <td>${escapeHtml(a.action)}</td>
    <td class="audit-summary">${escapeHtml(a.summary)}</td>
  </tr>`).join("");
  $("#auditBox").innerHTML = `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

// -- refresh ------------------------------------------------------------------

function refresh(panel) {
  if (panel === "dashboard") {
    loadDashboard().catch((e) => toast(e.message, "error"));
    loadChart().catch(() => {});
  }
  if (panel === "signals") loadSignals().catch((e) => toast(e.message, "error"));
  if (panel === "trades") loadTrades().catch((e) => toast(e.message, "error"));
  if (panel === "statistics") loadStatistics().catch((e) => toast(e.message, "error"));
  if (panel === "settings") loadSettings().catch((e) => toast(e.message, "error"));
}

refresh(current);
setInterval(() => { if (current === "dashboard") refresh("dashboard"); }, 15000);

$("#chartInterval").addEventListener("change", () => { chartView = null; loadChart(); });

["sigDir", "sigStatus", "sigTime"].forEach((id) => {
  $("#" + id).addEventListener("change", () => loadSignals().catch((e) => toast(e.message, "error")));
});

$("#signalsTable").addEventListener("change", (ev) => {
  if (ev.target.classList && ev.target.classList.contains("sig-check")) updateSigDeleteCount();
});

$("#sigSelAll").addEventListener("click", () => {
  const boxes = document.querySelectorAll("#signalsTable .sig-check");
  const allChecked = boxes.length > 0 && Array.from(boxes).every((b) => b.checked);
  boxes.forEach((b) => { b.checked = !allChecked; });
  $("#sigSelAll").textContent = allChecked ? "Select all" : "Deselect all";
  updateSigDeleteCount();
});

$("#sigDeleteSel").addEventListener("click", async () => {
  const ids = Array.from(document.querySelectorAll("#signalsTable .sig-check:checked"))
    .map((c) => Number(c.dataset.id));
  if (!ids.length) return;
  if (!confirm("Delete " + ids.length + " selected signal(s)?\nLinked demo positions/trades will also be deleted.")) return;
  try { await deleteSignals({ ids }); } catch (e) { toast(e.message, "error"); }
});

$("#sigDeleteAll").addEventListener("click", async () => {
  if (!confirm("Delete ALL signal history (except any OPEN position)?\nLinked demo positions/trades will also be deleted. This cannot be undone.")) return;
  try { await deleteSignals({ all: true }); } catch (e) { toast(e.message, "error"); }
});

let chartResizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(chartResizeTimer);
  chartResizeTimer = setTimeout(() => { if (current === "dashboard") loadChart().catch(() => {}); }, 300);
});