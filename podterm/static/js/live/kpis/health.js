import { kpiStore } from './store.js';

const BAND_LABEL = { ok: 'OK', warn: 'WARN', error: 'ERROR' };

function fmtHealthMetric(m) {
  if (m.value == null) return '—';
  const v = m.value;
  if (m.unit === 'frac') return `${(v * 100).toFixed(0)}%`;
  const a = Math.abs(v);
  const num = a !== 0 && (a < 0.01 || a >= 1000) ? v.toExponential(1) : String(Math.round(v * 100) / 100);
  return m.unit && m.unit !== 'frac' ? `${num}` : num;
}

function setVerdictClass(verdict) {
  const el = kpiStore.cards.health.el;
  el.classList.toggle('verdict-ok', verdict === 'ok');
  el.classList.toggle('verdict-warn', verdict === 'warn');
  el.classList.toggle('verdict-error', verdict === 'error');
}

export function updateHealthCard(state) {
  const { cards } = kpiStore;
  if (!cards) return;
  if (state.pendingSnapshotStep != null) {
    setVerdictClass(null);
    cards.health.set({ value: 'PENDING', sub: `snapshot pending · step ${state.pendingSnapshotStep}`,
                       caption: 'Open Health' });
    return;
  }
  const d0 = state.diagnostic;
  if (!d0 || !d0.health) { cards.health.note('No diagnostics yet…'); setVerdictClass(null); return; }
  const h = d0.health;
  const ok = h.overall === 'ok';
  setVerdictClass(h.overall);
  const offenders = h.headline.filter((m) => m.band === 'bad').concat(h.headline.filter((m) => m.band === 'warn'));
  const sub = offenders.length
    ? offenders.slice(0, 2).map((m) => `${m.label} ${fmtHealthMetric(m)}`).join(' · ')
    : `step ${d0.step} · ${h.counts.good || 0} ok`;
  cards.health.set({
    value: BAND_LABEL[h.overall] || '—',
    sub,
    subClass: ok ? 'success' : 'danger',
    caption: 'Open Health',
    accent: ok ? 'success' : 'danger',
  });
}
