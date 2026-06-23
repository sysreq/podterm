import { escapeHtml } from '../dom.js';

// ════════════════════════════════════════════════════════════════════════════
// Formatting helpers (shared across summary + deep dive)
// ════════════════════════════════════════════════════════════════════════════

export function bandClass(band) { return `diag-band-${band || 'na'}`; }

export function fmtVal(v, unit) {
  if (v == null) return '—';
  if (unit === 'frac') return `${(v * 100).toFixed(1)}%`;
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.001 || a >= 10000)) return v.toExponential(2);
  return String(Math.round(v * 1000) / 1000);
}

export function trendArrow(cur, prev) {
  if (prev == null || cur == null) return '';
  const d = cur - prev;
  if (Math.abs(d) < 1e-9 || (prev !== 0 && Math.abs(d / prev) < 1e-4)) return ' →';
  return d > 0 ? ` ▲${fmtVal(Math.abs(d))}` : ` ▼${fmtVal(Math.abs(d))}`;
}

export function fmtThreshold(t) {
  if (!t || t.kind === 'info') return '';
  if (t.kind === 'hi') return `warn ≥ ${fmtVal(t.warn)}${t.bad != null ? `, bad ≥ ${fmtVal(t.bad)}` : ''}`;
  if (t.kind === 'lo') return `warn ≤ ${fmtVal(t.warn)}${t.bad != null ? `, bad ≤ ${fmtVal(t.bad)}` : ''}`;
  if (t.kind === 'range') return `ok ${fmtVal(t.warn_lo)}–${fmtVal(t.warn_hi)}`;
  return '';
}

export function fmtAge(createdAt) {
  if (!createdAt) return '';
  const ms = Date.now() - Date.parse(createdAt);
  if (!isFinite(ms) || ms < 0) return '';
  const m = Math.floor(ms / 60000);
  if (m < 1) return 'just now';
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  return h < 24 ? `${h}h ${m % 60}m ago` : `${Math.floor(h / 24)}d ago`;
}

export const sevRank = (st) => (st === 'error' || st === 'bad' ? 2 : st === 'partial' || st === 'warn' ? 1 : 0);

export const VERDICT_LABEL = { ok: 'OK', warn: 'WARN', error: 'ERROR' };

// Section *execution* status — whether the probe ran to completion, distinct from
// the metric health verdict (good/warn/bad). Labelled "complete"/"partial" (not
// "ok") so a green badge here can't be misread as "metrics healthy" when the
// section actually carries warn-band metrics.
export const SECTION_STATUS_LABEL = { ok: 'complete', partial: 'partial', error: 'error', skipped: 'skipped' };
export const sectionStatusLabel = (st) => SECTION_STATUS_LABEL[st] || st || '';

export function makeEl(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

export function emptyState(title, msg) {
  const wrap = makeEl('div', 'health-empty');
  wrap.append(makeEl('div', 'health-empty-title', title), makeEl('div', 'health-empty-msg', msg));
  return wrap;
}


export function fmtRowMetrics(metrics, prev) {
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
