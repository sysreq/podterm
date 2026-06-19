// Model Health panel: renders the off-pod diagnostics time series for the active run.
//
// The producer (podterm/diagnostics/health.py) now curates the headline metrics and a per-snapshot
// verdict, so this panel renders that block directly — no more fragile substring matching. It also
// feeds the live verdict card (state.diagnostic + pod:health) and offers per-step / per-block
// drill-down with trend-vs-previous, all from the one history fetch.
import { app, emit, getPodState, on } from './state.js';
import { attachSparkline } from './sparkline.js';
import { escapeHtml } from './dom.js';
import { fetchJson } from './api.js';

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
    history = await fetchJson(`/api/runs/${runId}/diagnostics`);
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

  content.appendChild(renderVerdictBar(valid));
  content.appendChild(renderTimeline(valid, selIdx));
  content.appendChild(renderGroupedCards(valid));
  content.appendChild(renderDetail(valid, selIdx));
}

// ── verdict count bar: a proportional segmented bar from health.counts + a text legend ──
const COUNT_ORDER = ['good', 'warn', 'bad', 'info', 'na'];
function renderVerdictBar(valid) {
  const wrap = document.createElement('div');
  wrap.className = 'diag-verdict-bar';
  const counts = valid[valid.length - 1].diag.health?.counts || {};
  const total = COUNT_ORDER.reduce((a, b) => a + (counts[b] || 0), 0);
  const track = document.createElement('div');
  track.className = 'diag-vb-track';
  if (total) {
    for (const band of COUNT_ORDER) {
      const n = counts[band] || 0;
      if (!n) continue;
      const seg = document.createElement('span');
      seg.className = `diag-vb-seg diag-vb-${band}`;
      seg.style.flexGrow = String(n);
      seg.title = `${n} ${band}`;
      track.appendChild(seg);
    }
  }
  const legend = document.createElement('span');
  legend.className = 'diag-vb-legend';
  legend.textContent = COUNT_ORDER
    .filter((b) => counts[b])
    .map((b) => `${counts[b]} ${b}`)
    .join(' · ') || 'no metrics';
  wrap.append(track, legend);
  return wrap;
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

// ── producer-curated headline metrics as tiles grouped by concept (capacity/flow/grad/arch) ──
const GROUP_ORDER = ['capacity', 'flow', 'grad', 'arch'];
const GROUP_LABEL = { capacity: 'Capacity', flow: 'Flow', grad: 'Gradients', arch: 'Architecture' };

function renderGroupedCards(valid) {
  const wrap = document.createElement('div');
  wrap.className = 'diag-groups';
  const latest = valid[valid.length - 1].diag.health?.headline || [];
  const prev = valid.length > 1 ? (valid[valid.length - 2].diag.health?.headline || []) : [];
  const prevVal = (id) => prev.find((x) => x.id === id)?.value;

  // Fixed group order first, then any unexpected groups appended so nothing silently vanishes.
  const seen = new Set(GROUP_ORDER);
  const groups = GROUP_ORDER.concat([...new Set(latest.map((m) => m.group))].filter((g) => !seen.has(g)));

  for (const g of groups) {
    const metrics = latest.filter((m) => m.group === g);
    if (!metrics.length) continue;

    const head = document.createElement('div');
    head.className = 'diag-group';
    head.textContent = GROUP_LABEL[g] || g || 'Other';
    wrap.appendChild(head);

    const grid = document.createElement('div');
    grid.className = 'diag-cards';
    for (const m of metrics) {
      const series = valid
        .map((h) => (h.diag.health?.headline || []).find((x) => x.id === m.id)?.value)
        .filter((v) => v != null);
      const card = document.createElement('div');
      card.className = 'diag-card';

      const label = document.createElement('div');
      label.className = 'diag-card-label';
      label.textContent = m.label;

      const valLine = document.createElement('div');
      valLine.className = 'diag-card-valline';
      const val = document.createElement('span');
      val.className = `diag-card-val ${bandClass(m.band)}`;
      val.textContent = fmtVal(m.value, m.unit);
      const trend = document.createElement('span');
      trend.className = 'diag-card-trend';
      trend.textContent = trendArrow(m.value, prevVal(m.id));
      valLine.append(val, trend);

      const spark = document.createElement('div');
      spark.className = 'diag-card-spark';

      card.append(label, valLine, spark);
      grid.appendChild(card);
      if (series.length > 1) attachSparkline(spark).update(series);
    }
    wrap.appendChild(grid);
  }
  return wrap;
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

  // Worst-first so problems sort to the top; sections are open by default (data not hidden).
  const sevRank = (st) => (st === 'error' ? 2 : st === 'partial' ? 1 : 0);
  const sections = [...(cur.sections || [])].sort((a, b) => sevRank(b.status) - sevRank(a.status));

  for (const s of sections) {
    const sec = document.createElement('details');
    sec.className = 'diag-section';
    sec.open = true;
    const summary = document.createElement('summary');
    summary.innerHTML = `<span class="diag-sec-name">${escapeHtml(s.name)}</span>` +
      `<span class="diag-sec-status diag-${escapeHtml(s.status)}">${escapeHtml(s.status)}${s.reason ? ': ' + escapeHtml(s.reason) : ''}</span>`;
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
      parts.push(`<span class="diag-kv"><b>${escapeHtml(k)}</b> ${escapeHtml(fmtVal(v))}<span class="diag-trend">${escapeHtml(trendArrow(v, pv))}</span></span>`);
    } else if (Array.isArray(v) && v.length && v.every((x) => typeof x === 'number')) {
      const mean = v.reduce((a, b) => a + b, 0) / v.length;
      parts.push(`<span class="diag-kv"><b>${escapeHtml(k)}</b> [${v.length}] μ${escapeHtml(fmtVal(mean))} <span class="diag-trend">min ${escapeHtml(fmtVal(Math.min(...v)))} max ${escapeHtml(fmtVal(Math.max(...v)))}</span></span>`);
    } else if (typeof v === 'string') {
      parts.push(`<span class="diag-kv"><b>${escapeHtml(k)}</b> ${escapeHtml(v)}</span>`);
    }
  }
  if (metrics._unverified) parts.push('<span class="diag-unverified">unverified</span>');
  return parts.join(' ');
}
