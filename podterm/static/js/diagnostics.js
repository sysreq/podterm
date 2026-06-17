// Model Health panel: renders the off-pod diagnostics time series for the active run.
// Each snapshot is a full schema-v2 diagnostics doc; we flatten numeric metrics and trend a
// curated set of headline health indicators across steps. Falls back to raw JSON if none match.
import { app, on } from './state.js';
import { attachSparkline } from './sparkline.js';

// [label, section-substring, key-substring, aggregate] — matched case-insensitively across the
// per-block rows of each snapshot. 'max' surfaces the worst block; 'mean' averages them.
const HEADLINE = [
  ['Dead neurons',        'capacity', 'dead',     'max'],
  ['Head redundancy',     'redundan', 'max',      'max'],
  ['Attn entropy (bits)', 'entropy',  'bit',      'mean'],
  ['Q-sim (mean)',        'similar',  'mean',     'mean'],
  ['Zero-init escape',    'escape',   '',         'max'],
  ['Embed eff_rank',      'spectrum', 'eff_rank', 'mean'],
];

const el = (id) => document.getElementById(id);

function flatten(diag) {
  const out = [];
  for (const s of (diag?.sections || [])) {
    for (const [, metrics] of Object.entries(s.rows || {})) {
      if (metrics && typeof metrics === 'object') {
        for (const [k, v] of Object.entries(metrics)) {
          if (typeof v === 'number' && isFinite(v)) {
            out.push({ section: (s.name || '').toLowerCase(), key: k.toLowerCase(), value: v });
          }
        }
      }
    }
  }
  return out;
}

function aggregate(flat, secMatch, keyMatch, how) {
  const vals = flat
    .filter((f) => f.section.includes(secMatch) && f.key.includes(keyMatch))
    .map((f) => f.value);
  if (!vals.length) return null;
  return how === 'max' ? Math.max(...vals) : vals.reduce((a, b) => a + b, 0) / vals.length;
}

function fmt(v) {
  if (v == null) return '—';
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.001 || a >= 10000)) return v.toExponential(2);
  return String(Math.round(v * 1000) / 1000);
}

export function initDiagnostics() {
  on('pod:hydrated', ({ podId }) => { if (podId === app.activePod) load(podId); });
  on('pod:diagnostic', ({ podId }) => { if (podId === app.activePod) load(podId); });
  on('pod:reset', () => { const p = el('diagnostics-panel'); if (p) p.hidden = true; });
}

async function load(runId) {
  let history;
  try {
    history = await fetch(`/api/runs/${runId}/diagnostics`).then((r) => r.json());
  } catch {
    return;
  }
  if (app.activePod !== runId) return; // user switched during the fetch
  render(Array.isArray(history) ? history : []);
}

function render(history) {
  const panel = el('diagnostics-panel');
  const content = el('diag-content');
  const sub = el('diag-sub');
  const valid = history.filter((h) => h && h.diag);
  if (!valid.length) { panel.hidden = true; return; }
  panel.hidden = false;

  const latest = valid[valid.length - 1];
  sub.textContent = `${valid.length} snapshot${valid.length > 1 ? 's' : ''} · latest step ${latest.step} · ${latest.status || '?'}`;
  content.innerHTML = '';

  // Status timeline — one chip per snapshot step, coloured by overall status.
  const timeline = document.createElement('div');
  timeline.className = 'diag-timeline';
  for (const h of valid) {
    const chip = document.createElement('span');
    chip.className = `diag-chip diag-${h.status || 'unknown'}`;
    chip.textContent = h.step;
    chip.title = `step ${h.step} · ${h.status || 'unknown'}${h.created_at ? ' · ' + h.created_at : ''}`;
    timeline.appendChild(chip);
  }
  content.appendChild(timeline);

  // Headline metric trends across steps.
  const flats = valid.map((h) => flatten(h.diag));
  const table = document.createElement('div');
  table.className = 'diag-metrics';
  let any = false;
  for (const [label, secM, keyM, how] of HEADLINE) {
    const series = flats.map((f) => aggregate(f, secM, keyM, how));
    if (series.every((v) => v == null)) continue;
    any = true;
    const row = document.createElement('div');
    row.className = 'diag-metric-row';
    const name = document.createElement('span');
    name.className = 'diag-metric-label';
    name.textContent = label;
    const val = document.createElement('span');
    val.className = 'diag-metric-val';
    val.textContent = fmt(series[series.length - 1]);
    const spark = document.createElement('span');
    spark.className = 'diag-metric-spark';
    row.append(name, val, spark);
    table.appendChild(row);
    attachSparkline(spark).update(series.filter((v) => v != null));
  }
  content.appendChild(table);

  // Nothing matched the curated set (e.g. weights-only run on CPU): show the raw doc.
  if (!any) {
    const det = document.createElement('details');
    det.className = 'diag-raw';
    const summary = document.createElement('summary');
    summary.textContent = 'Latest diagnostics JSON';
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(latest.diag, null, 2);
    det.append(summary, pre);
    content.appendChild(det);
  }
}
