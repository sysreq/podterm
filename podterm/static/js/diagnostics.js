// Model Health panel: renders the off-pod diagnostics time series for the active run.
//
// The producer (podterm/diagnostics/health.py) now curates the headline metrics and a per-snapshot
// verdict, so this panel renders that block directly — no more fragile substring matching. It also
// feeds the live verdict card (state.diagnostic + pod:health) and offers per-step / per-block
// drill-down with trend-vs-previous, all from the one history fetch.
import { app, emit, getPodState, on } from './state.js';
import { attachSparkline } from './sparkline.js';

const el = (id) => document.getElementById(id);
let selectedStep = null; // which snapshot the detail view is pinned to (null = latest)

export function initDiagnostics() {
  on('pod:hydrated', ({ podId }) => { if (podId === app.activePod) load(podId); });
  on('pod:diagnostic', ({ podId }) => { if (podId === app.activePod) load(podId); });
  on('pod:reset', () => { const p = el('diagnostics-panel'); if (p) p.hidden = true; selectedStep = null; });
}

async function load(runId) {
  let history;
  try {
    history = await fetch(`/api/runs/${runId}/diagnostics`).then((r) => r.json());
  } catch {
    return;
  }
  if (app.activePod !== runId) return; // user switched during the fetch
  const valid = (Array.isArray(history) ? history : []).filter((h) => h && h.diag);

  // Keep the live verdict card in sync from the same fetch (covers reconnects / page loads where no
  // fresh SSE 'diagnostic' has arrived yet).
  const st = getPodState(runId);
  if (valid.length) {
    const l = valid[valid.length - 1];
    st.diagnostic = { step: l.step, status: l.status, health: l.diag.health || null };
    emit('pod:health', { podId: runId });
  }
  render(valid);
}

// ── band → css class + value formatting ──
function bandClass(band) { return `diag-band-${band || 'na'}`; }

function fmtVal(v, unit) {
  if (v == null) return '—';
  if (unit === 'frac') return `${(v * 100).toFixed(1)}%`;
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.001 || a >= 10000)) return v.toExponential(2);
  return String(Math.round(v * 1000) / 1000);
}

function trendArrow(cur, prev) {
  if (prev == null || cur == null) return '';
  const d = cur - prev;
  if (Math.abs(d) < 1e-9 || (prev !== 0 && Math.abs(d / prev) < 1e-4)) return ' →';
  return d > 0 ? ` ▲${fmtVal(Math.abs(d))}` : ` ▼${fmtVal(Math.abs(d))}`;
}

function render(valid) {
  const panel = el('diagnostics-panel');
  const content = el('diag-content');
  const sub = el('diag-sub');
  if (!valid.length) { panel.hidden = true; return; }
  panel.hidden = false;

  const latest = valid[valid.length - 1];
  const verdict = latest.diag.health?.overall || latest.status || '?';
  sub.textContent = `${valid.length} snapshot${valid.length > 1 ? 's' : ''} · latest step ${latest.step} · ${verdict}`;
  content.innerHTML = '';

  // Resolve which snapshot the detail view shows (pinned step, else latest).
  let selIdx = valid.length - 1;
  if (selectedStep != null) {
    const i = valid.findIndex((h) => h.step === selectedStep);
    if (i >= 0) selIdx = i;
  }

  content.appendChild(renderTimeline(valid, selIdx));
  content.appendChild(renderHeadline(valid));
  content.appendChild(renderDetail(valid, selIdx));
}

// ── status timeline (clickable: pins the detail view to a step) ──
function renderTimeline(valid, selIdx) {
  const timeline = document.createElement('div');
  timeline.className = 'diag-timeline';
  valid.forEach((h, i) => {
    const chip = document.createElement('button');
    const verdict = h.diag.health?.overall || h.status || 'unknown';
    chip.className = `diag-chip diag-${verdict}` + (i === selIdx ? ' selected' : '');
    chip.textContent = h.step;
    chip.title = `step ${h.step} · ${verdict}${h.created_at ? ' · ' + h.created_at : ''}`;
    chip.addEventListener('click', () => { selectedStep = h.step; render(valid); });
    timeline.appendChild(chip);
  });
  return timeline;
}

// ── producer-curated headline metrics with threshold colour + cross-snapshot sparkline ──
function renderHeadline(valid) {
  const table = document.createElement('div');
  table.className = 'diag-metrics';
  const headline = valid[valid.length - 1].diag.health?.headline || [];
  for (const m of headline) {
    const series = valid
      .map((h) => (h.diag.health?.headline || []).find((x) => x.id === m.id)?.value)
      .filter((v) => v != null);
    const row = document.createElement('div');
    row.className = 'diag-metric-row';
    const name = document.createElement('span');
    name.className = 'diag-metric-label';
    name.textContent = m.label;
    const val = document.createElement('span');
    val.className = `diag-metric-val ${bandClass(m.band)}`;
    val.textContent = fmtVal(m.value, m.unit);
    const spark = document.createElement('span');
    spark.className = 'diag-metric-spark';
    row.append(name, val, spark);
    table.appendChild(row);
    if (series.length > 1) attachSparkline(spark).update(series);
  }
  return table;
}

// ── per-section / per-block detail for the selected snapshot, with trend vs the previous one ──
function renderDetail(valid, selIdx) {
  const wrap = document.createElement('div');
  wrap.className = 'diag-detail';
  const cur = valid[selIdx].diag;
  const prev = selIdx > 0 ? valid[selIdx - 1].diag : null;
  const prevSection = (name) => (prev?.sections || []).find((s) => s.name === name);

  const head = document.createElement('div');
  head.className = 'diag-detail-head';
  head.textContent = `Step ${valid[selIdx].step} — per-section detail${prev ? ` (▲▼ vs step ${valid[selIdx - 1].step})` : ''}`;
  wrap.appendChild(head);

  for (const s of (cur.sections || [])) {
    const sec = document.createElement('details');
    sec.className = 'diag-section';
    // Open the sections that aren't clean so problems are visible without a click.
    sec.open = s.status === 'error' || s.status === 'partial';
    const summary = document.createElement('summary');
    summary.innerHTML = `<span class="diag-sec-name">${s.name}</span>` +
      `<span class="diag-sec-status diag-${s.status}">${s.status}${s.reason ? ': ' + s.reason : ''}</span>`;
    sec.appendChild(summary);

    const rows = s.rows || {};
    const prevRows = prevSection(s.name)?.rows || {};
    if (Object.keys(rows).length) {
      const table = document.createElement('table');
      table.className = 'diag-rows';
      for (const [rk, metrics] of Object.entries(rows)) {
        const tr = document.createElement('tr');
        const td0 = document.createElement('td');
        td0.className = 'diag-row-key';
        td0.textContent = rk;
        const td1 = document.createElement('td');
        td1.innerHTML = fmtRowMetrics(metrics, prevRows[rk]);
        tr.append(td0, td1);
        table.appendChild(tr);
      }
      sec.appendChild(table);
    }
    wrap.appendChild(sec);
  }

  // Raw JSON kept available for the truly curious.
  const raw = document.createElement('details');
  raw.className = 'diag-raw';
  const rs = document.createElement('summary');
  rs.textContent = 'Raw diagnostics JSON';
  const pre = document.createElement('pre');
  pre.textContent = JSON.stringify(cur, null, 2);
  raw.append(rs, pre);
  wrap.appendChild(raw);
  return wrap;
}

function fmtRowMetrics(metrics, prev) {
  if (!metrics || typeof metrics !== 'object') return '';
  const parts = [];
  for (const [k, v] of Object.entries(metrics)) {
    if (k === '_unverified') continue;
    if (typeof v === 'number' && isFinite(v)) {
      const pv = prev && typeof prev[k] === 'number' ? prev[k] : null;
      parts.push(`<span class="diag-kv"><b>${k}</b> ${fmtVal(v)}<span class="diag-trend">${trendArrow(v, pv)}</span></span>`);
    } else if (Array.isArray(v) && v.length && v.every((x) => typeof x === 'number')) {
      const mean = v.reduce((a, b) => a + b, 0) / v.length;
      parts.push(`<span class="diag-kv"><b>${k}</b> [${v.length}] μ${fmtVal(mean)} <span class="diag-trend">min ${fmtVal(Math.min(...v))} max ${fmtVal(Math.max(...v))}</span></span>`);
    } else if (typeof v === 'string') {
      parts.push(`<span class="diag-kv"><b>${k}</b> ${v}</span>`);
    }
  }
  if (metrics._unverified) parts.push('<span class="diag-unverified">unverified</span>');
  return parts.join(' ');
}
