// Compare tab: overlay charts (step- or progress-aligned), regression flagging, a Model Health diff
// (verdict + headline side-by-side + per-block movers), and a summary table.
import { app, emit } from './state.js';
import { scaleY, lossYAxis, lossHover, plotLayout, plotConfig, COLORS } from './charts.js';
import { observePlotlyResize } from './resize.js';
import { escapeHtml } from './dom.js';
import { fetchJson } from './api.js';

let lastCompareData = null; // cached for re-render on chip removal / align toggle
let alignMode = 'step';     // 'step' | 'progress'

export async function compareSelected() {
  if (app.selectedRuns.size < 2) { alert('Select at least 2 runs.'); return; }
  emit('tab:switch', { tab: 'compare' });

  try {
    lastCompareData = await fetchJson('/api/compare', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ run_ids: [...app.selectedRuns] }),
    });
    renderComparison();
  } catch (e) {
    document.getElementById('compare-placeholder').style.display = '';
    document.getElementById('compare-placeholder').textContent = `Compare failed: ${e.message || e}`;
  }
}

function removeFromComparison(runId) {
  app.selectedRuns.delete(runId);
  document.getElementById('selection-count').textContent = `${app.selectedRuns.size} selected`;
  if (app.selectedRuns.size < 2) {
    document.getElementById('compare-selection').style.display = 'none';
    for (const id of ['compare-regression', 'compare-controls', 'compare-charts', 'compare-config', 'compare-health', 'compare-summary']) {
      document.getElementById(id).innerHTML = '';
    }
    document.getElementById('compare-placeholder').style.display = '';
    document.getElementById('compare-placeholder').textContent = 'Select at least 2 runs from the History tab.';
    lastCompareData = null;
    import('./history.js').then((m) => m.renderHistoryTable());
    return;
  }
  if (lastCompareData) {
    delete lastCompareData.metrics[runId];
    delete lastCompareData.runs[runId];
    if (lastCompareData.diagnostics) {
      delete lastCompareData.diagnostics.health?.[runId];
      delete lastCompareData.diagnostics.diff?.[runId];
    }
    renderComparison();
  }
  import('./history.js').then((m) => m.renderHistoryTable());
}

const runLabel = (run) => `${run.branch || ''}@${(run.commit_hash || '').slice(0, 7)}`;

function renderComparison() {
  const data = lastCompareData;
  if (!data) return;
  document.getElementById('compare-placeholder').style.display = 'none';

  // Selection chips
  const selEl = document.getElementById('compare-selection');
  selEl.style.display = 'flex';
  let chipsHtml = '<span>Comparing:</span>';
  for (const rid of Object.keys(data.runs)) {
    const label = runLabel(data.runs[rid] || {});
    chipsHtml += `<div class="run-chip"><span>${escapeHtml(label)}</span><span class="chip-x" data-run-id="${escapeHtml(rid)}">&times;</span></div>`;
  }
  selEl.innerHTML = chipsHtml;

  renderRegression(data);
  renderControls();
  renderCharts(data);
  renderConfigRepro(data);
  renderHealthDiff(data);
  renderSummary(data);
}

// ── alignment toggle ──
function renderControls() {
  const el = document.getElementById('compare-controls');
  el.innerHTML =
    '<span class="cmp-ctl-label">Align X by:</span>' +
    `<button class="cmp-toggle${alignMode === 'step' ? ' active' : ''}" data-align="step">Step</button>` +
    `<button class="cmp-toggle${alignMode === 'progress' ? ' active' : ''}" data-align="progress">Progress %</button>`;
  el.querySelectorAll('.cmp-toggle').forEach((b) =>
    b.addEventListener('click', () => { alignMode = b.dataset.align; renderComparison(); }));
}

function alignX(steps) {
  if (alignMode !== 'progress' || !steps.length) return steps;
  const max = steps[steps.length - 1] || 1;
  return steps.map((s) => s / max);
}

// ── regression banner: final train-loss vs the base run ──
function renderRegression(data) {
  const el = document.getElementById('compare-regression');
  el.innerHTML = '';
  const ids = Object.keys(data.metrics);
  const finalLoss = (rid) => {
    const ms = (data.metrics[rid] || []).filter((m) => m.train_loss > 0);
    return ms.length ? ms[ms.length - 1].train_loss : null;
  };
  const baseId = data.diagnostics?.base || ids[0];
  const baseLoss = finalLoss(baseId);
  if (baseLoss == null) return;
  const flags = [];
  for (const rid of ids) {
    if (rid === baseId) continue;
    const l = finalLoss(rid);
    if (l == null) continue;
    const pct = ((l - baseLoss) / baseLoss) * 100;
    if (pct > 1) flags.push(`${runLabel(data.runs[rid] || {})} is ${pct.toFixed(1)}% worse on final loss`);
  }
  if (flags.length) {
    el.className = 'cmp-banner danger';
    el.textContent = '⚠ Regression vs ' + runLabel(data.runs[baseId] || {}) + ': ' + flags.join(' · ');
  } else {
    el.className = 'cmp-banner ok';
    el.textContent = '✓ No final-loss regression vs ' + runLabel(data.runs[baseId] || {});
  }
}

function renderCharts(data) {
  // Only the training-loss curve earns a full chart — it's a dense series. Validation
  // BPB is one (or a few) end-of-run points, so it's a compact readout in the
  // Config / Repro panel below, not a near-empty 300px plot.
  const lossTraces = [];
  let i = 0;
  for (const [rid, metrics] of Object.entries(data.metrics)) {
    const label = runLabel(data.runs[rid] || {});
    const color = COLORS[i % COLORS.length];
    const cmpMetrics = metrics.filter((m) => m.train_loss);
    const trainSteps = cmpMetrics.map((m) => m.step);
    if (trainSteps.length) {
      lossTraces.push({ x: alignX(trainSteps), y: cmpMetrics.map((m) => scaleY(m.train_loss)),
        customdata: cmpMetrics.map((m) => m.train_loss), name: label, type: 'scatter', line: { color }, hovertemplate: lossHover });
    }
    i++;
  }
  const xTitle = alignMode === 'progress' ? 'Progress (fraction of run)' : 'Step';
  const chartsEl = document.getElementById('compare-charts');
  chartsEl.innerHTML = '<div class="chart-box" id="cmp-loss"></div>';
  Plotly.newPlot('cmp-loss', lossTraces, { ...plotLayout, yaxis: lossYAxis, xaxis: { title: xTitle }, title: 'Training Loss Comparison', height: 300 }, plotConfig);
  observePlotlyResize(document.getElementById('cmp-loss'));
}

// ── Validation BPB readout + Config / Repro side-by-side ──
// Repro-relevant fields pulled from the run row (r) and its parsed launch config (c).
const REPRO_FIELDS = [
  ['Commit', (r) => r.commit_hash || '—'],
  ['Branch', (r) => r.branch || '—'],
  ['Seed', (r) => (r.seed != null ? String(r.seed) : '—')],
  ['Data variant', (r, c) => r.data_variant || c.data_variant || '—'],
  ['Vocab size', (r, c) => (r.vocab_size ?? c.vocab_size ?? '—').toString()],
  ['Seq len', (r) => (r.seq_len != null ? String(r.seq_len) : '—')],
  ['Batch tokens', (r) => (r.batch_tokens != null ? r.batch_tokens.toLocaleString() : '—')],
  ['Time budget', (r, c) => (c.time_budget != null ? `${c.time_budget}s` : '—')],
  ['Steps', (r) => (r.total_steps != null ? r.total_steps.toLocaleString() : '—')],
  ['Model params', (r) => (r.model_params != null ? `${(r.model_params / 1e6).toFixed(2)}M` : '—')],
  ['GPU', (r) => (r.gpu_type ? `${r.gpu_count ? r.gpu_count + '× ' : ''}${r.gpu_type}` : '—')],
  ['Train script', (r, c) => c.train_script || '—'],
];

function renderConfigRepro(data) {
  const el = document.getElementById('compare-config');
  const ids = Object.keys(data.runs);
  const baseId = data.diagnostics?.base || ids[0];
  const cfg = {};
  for (const rid of ids) {
    try { cfg[rid] = JSON.parse(data.runs[rid]?.config_json || '{}') || {}; } catch { cfg[rid] = {}; }
  }

  // Compact Validation BPB: best (lowest) per run, winner highlighted.
  const bpb = (rid) => data.runs[rid]?.best_val_bpb;
  const present = ids.map(bpb).filter((v) => v != null);
  const best = present.length ? Math.min(...present) : null;
  let html = '<div class="cmp-section-title">Validation BPB (best)</div><div class="cmp-bpb-strip">';
  for (const rid of ids) {
    const v = bpb(rid);
    const isBest = v != null && best != null && Math.abs(v - best) < 1e-9 && ids.length > 1;
    html += `<div class="cmp-bpb-stat${isBest ? ' best' : ''}"><span class="cmp-bpb-run">${escapeHtml(runLabel(data.runs[rid] || {}))}</span>`
      + `<span class="cmp-bpb-val">${v != null ? v.toFixed(4) : '—'}</span></div>`;
  }
  html += '</div>';

  // Config / Repro matrix: field (rows) × run (cols); rows that differ are flagged.
  html += '<div class="cmp-section-title">Config / Repro</div><table class="cmp-config"><thead><tr><th>Field</th>';
  for (const rid of ids) html += `<th>${escapeHtml(runLabel(data.runs[rid] || {}))}${rid === baseId ? ' <span class="cmp-base">base</span>' : ''}</th>`;
  html += '</tr></thead><tbody>';
  for (const [label, get] of REPRO_FIELDS) {
    const vals = ids.map((rid) => String(get(data.runs[rid] || {}, cfg[rid] || {})));
    const differ = vals.some((v) => v !== vals[0]);
    html += `<tr${differ ? ' class="cmp-diff"' : ''}><td>${escapeHtml(label)}</td>`;
    for (const v of vals) html += `<td>${escapeHtml(v)}</td>`;
    html += '</tr>';
  }
  html += '</tbody></table>';
  el.innerHTML = html;
}

// ── Model Health diff: verdict + headline side-by-side, plus per-run movers vs base ──
const VERDICT_LABEL = { ok: 'OK', warn: 'WARN', error: 'ERROR' };
const fmtHeadline = (m) => {
  if (m == null || m.value == null) return '—';
  if (m.unit === 'frac') return `${(m.value * 100).toFixed(1)}%`;
  const a = Math.abs(m.value);
  return a !== 0 && (a < 0.001 || a >= 10000) ? m.value.toExponential(2) : String(Math.round(m.value * 1000) / 1000);
};

function renderHealthDiff(data) {
  const el = document.getElementById('compare-health');
  el.innerHTML = '';
  const diag = data.diagnostics;
  const ids = Object.keys(data.runs);
  const anyHealth = diag && diag.health && ids.some((r) => diag.health[r]);
  if (!anyHealth) {
    el.innerHTML = '<div class="cmp-section-title">Model Health</div><div class="cmp-empty">No diagnostics recorded for the selected runs.</div>';
    return;
  }
  const baseId = diag.base;

  // Headline matrix: metric label (rows) × run (cols).
  const order = (diag.health[baseId]?.headline || diag.health[ids.find((r) => diag.health[r])].headline || []).map((m) => m.id);
  const byRun = {};
  for (const rid of ids) {
    byRun[rid] = {};
    for (const m of (diag.health[rid]?.headline || [])) byRun[rid][m.id] = m;
  }
  const labelOf = (id) => {
    for (const rid of ids) if (byRun[rid][id]) return byRun[rid][id].label;
    return id;
  };

  let html = '<div class="cmp-section-title">Model Health</div><table class="cmp-health"><thead><tr><th>Metric</th>';
  for (const rid of ids) html += `<th>${escapeHtml(runLabel(data.runs[rid] || {}))}${rid === baseId ? ' <span class="cmp-base">base</span>' : ''}</th>`;
  html += '</tr></thead><tbody>';
  // Verdict row
  html += '<tr class="cmp-verdict-row"><td>Verdict</td>';
  for (const rid of ids) {
    const v = diag.health[rid]?.overall;
    html += `<td><span class="diag-band-${v === 'ok' ? 'good' : v === 'warn' ? 'warn' : v === 'error' ? 'bad' : 'na'}">${VERDICT_LABEL[v] || '—'}</span></td>`;
  }
  html += '</tr>';
  // Metric rows
  for (const id of order) {
    html += `<tr><td>${escapeHtml(labelOf(id))}</td>`;
    for (const rid of ids) {
      const m = byRun[rid][id];
      html += `<td><span class="diag-band-${m?.band || 'na'}">${fmtHeadline(m)}</span></td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';

  // Per-block movers vs base (the structured diff_docs output).
  if (diag.diff) {
    for (const rid of ids) {
      if (rid === baseId || !diag.diff[rid]) continue;
      const movers = (diag.diff[rid].movers || []).slice(0, 10);
      if (!movers.length) continue;
      html += `<details class="cmp-movers"><summary>Top changes: ${escapeHtml(runLabel(data.runs[rid] || {}))} vs base (${movers.length})</summary><table class="cmp-health"><thead><tr><th>Metric</th><th>base → run (Δ)</th></tr></thead><tbody>`;
      for (const mv of movers) html += `<tr><td class="diag-row-key">${escapeHtml(mv.path)}</td><td>${escapeHtml(mv.text)}</td></tr>`;
      html += '</tbody></table></details>';
    }
  }
  el.innerHTML = html;
}

function renderSummary(data) {
  const diag = data.diagnostics || {};
  let html = '<table><thead><tr><th>Run</th><th>Branch</th><th>GPU</th><th>Steps</th><th>Best BPB</th><th>Health</th><th>Cost</th></tr></thead><tbody>';
  for (const rid of Object.keys(data.runs)) {
    const run = data.runs[rid] || {};
    const bpb = run.best_val_bpb != null ? run.best_val_bpb.toFixed(4) : '-';
    const cost = run.total_cost != null ? `$${run.total_cost.toFixed(3)}` : '-';
    const v = diag.health?.[rid]?.overall;
    const health = v
      ? `<span class="diag-band-${v === 'ok' ? 'good' : v === 'warn' ? 'warn' : 'bad'}">${VERDICT_LABEL[v]}</span>`
      : '-';
    html += `<tr><td>${escapeHtml((run.pod_name || rid).slice(0, 20))}</td><td>${escapeHtml(run.branch || '-')}</td><td>${escapeHtml(run.gpu_type || '-')}</td><td>${escapeHtml(run.total_steps || '-')}</td><td>${escapeHtml(bpb)}</td><td>${health}</td><td>${escapeHtml(cost)}</td></tr>`;
  }
  html += '</tbody></table>';
  document.getElementById('compare-summary').innerHTML = html;
}

export function initCompare() {
  document.getElementById('compare-selection').addEventListener('click', (e) => {
    const x = e.target.closest('.chip-x');
    if (x) removeFromComparison(x.dataset.runId);
  });
}
