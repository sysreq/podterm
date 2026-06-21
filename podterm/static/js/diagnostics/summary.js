import { getPodState } from '../state.js';
import {
  BAND_ORDER, groupSummary, headlineOf, previousSnapshot, topHealthIssues,
} from './selectors.js';
import {
  VERDICT_LABEL, bandClass, fmtAge, fmtThreshold, fmtVal, makeEl, trendArrow,
} from './format.js';


function interpretation(snapshot) {
  const h = (snapshot.diag && snapshot.diag.health) || {};
  if ((h.overall || '') === 'ok') return 'All model-health checks look healthy.';
  const failing = groupSummary(snapshot).filter((g) => g.worst === 'warn' || g.worst === 'bad');
  const counts = h.counts || {};
  const issues = (counts.bad || 0) + (counts.warn || 0);
  if (!failing.length || !issues) return 'Some checks need review.';
  const names = failing.map((g) => g.label.toLowerCase());
  const joined = names.length === 1 ? names[0]
    : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
  return `${issues} ${issues === 1 ? 'check needs' : 'checks need'} review in ${joined}.`;
}

export function renderHealthSummary(ctx) {
  const wrap = makeEl('div', 'health-summary');
  const snap = ctx.currentSnapshot();
  const h = (snap.diag && snap.diag.health) || {};
  const overall = h.overall || snap.status || 'na';
  const st = getPodState(ctx.state.activeRunId);
  const run = st.runRow || {};

  // Header bar: identity + latest snapshot + age + verdict + deep-dive entry.
  const header = makeEl('div', 'health-header');
  const idText = [run.pod_name || run.name || ctx.state.activeRunId, run.branch, run.commit_hash ? run.commit_hash.slice(0, 7) : null]
    .filter(Boolean).join(' · ');
  const left = makeEl('div', 'health-header-id', idText);
  const meta = makeEl('div', 'health-header-meta');
  meta.append(makeEl('span', null, `step ${snap.step}`), makeEl('span', null, fmtAge(snap.created_at) || '—'));
  const right = makeEl('div', 'health-header-actions');
  const badge = makeEl('span', `health-badge ${bandClass(overall === 'ok' ? 'good' : overall === 'warn' ? 'warn' : 'bad')}`,
    VERDICT_LABEL[overall] || String(overall).toUpperCase());
  const deepBtn = makeEl('button', 'btn-ghost', 'Deep Dive');
  deepBtn.addEventListener('click', () => { ctx.state.view.mode = 'deep'; ctx.renderHealthTab(); });
  right.append(badge, deepBtn);
  header.append(left, meta, right);
  wrap.appendChild(header);

  // Verdict band: big verdict + interpretation + the five counts.
  const band = makeEl('div', `health-hero ${bandClass(overall === 'ok' ? 'good' : overall === 'warn' ? 'warn' : 'bad')}`);
  band.appendChild(makeEl('div', 'health-verdict', VERDICT_LABEL[overall] || String(overall).toUpperCase()));
  band.appendChild(makeEl('div', 'health-interp', interpretation(snap)));
  const counts = h.counts || {};
  const countWrap = makeEl('div', 'health-counts');
  const COUNT_VIEW = [['good', 'healthy'], ['warn', 'warnings'], ['bad', 'failing'], ['na', 'unavailable'], ['info', 'info']];
  for (const [k, label] of COUNT_VIEW) {
    const c = makeEl('span', `health-count diag-band-${k === 'good' ? 'good' : k === 'warn' ? 'warn' : k === 'bad' ? 'bad' : k}`);
    c.append(makeEl('b', null, String(counts[k] || 0)), document.createTextNode(' ' + label));
    countWrap.appendChild(c);
  }
  band.appendChild(countWrap);

  // Top issues vs the rest.
  const cols = makeEl('div', 'health-cols');
  cols.append(band, renderTopIssues(ctx, snap));
  wrap.appendChild(cols);

  wrap.appendChild(renderGroups(snap));
  wrap.appendChild(renderSnapshotTimeline(ctx));
  return wrap;
}

function renderTopIssues(ctx, snap) {
  const box = makeEl('div', 'health-issues');
  const issues = topHealthIssues(snap);
  const healthy = issues.length && issues.every((m) => m.band === 'good');
  box.appendChild(makeEl('div', 'health-issues-title', healthy ? 'Healthiest metrics' : 'Needs attention'));
  if (!issues.length) { box.appendChild(makeEl('div', 'health-empty-msg', 'No headline metrics available.')); return box; }
  const prev = previousSnapshot(ctx.state.history, snap.step);
  const prevVal = (id) => (headlineOf(prev).find((x) => x.id === id) || {}).value;
  for (const m of issues) {
    const row = makeEl('div', 'health-issue');
    const head = makeEl('div', 'health-issue-head');
    head.appendChild(makeEl('span', 'health-issue-label', m.label));
    const val = makeEl('span', `health-issue-val ${bandClass(m.band)}`, fmtVal(m.value, m.unit));
    val.appendChild(makeEl('span', 'health-issue-trend', trendArrow(m.value, prevVal(m.id))));
    head.appendChild(val);
    row.appendChild(head);
    const hint = [m.why, fmtThreshold(m.thresholds)].filter(Boolean).join(' · ');
    if (hint) row.appendChild(makeEl('div', 'health-issue-why', hint));
    box.appendChild(row);
  }
  return box;
}

function renderGroups(snap) {
  const wrap = makeEl('div', 'health-groups');
  for (const g of groupSummary(snap)) {
    const row = makeEl('div', 'health-group-row');
    const head = makeEl('div', 'health-group-head');
    head.appendChild(makeEl('span', `health-badge ${bandClass(g.worst)}`, g.label));
    const tally = BAND_ORDER.filter((b) => g.counts[b]).map((b) => `${g.counts[b]} ${b}`).join(' · ');
    head.appendChild(makeEl('span', 'health-group-tally', tally));
    row.appendChild(head);
    const vals = makeEl('div', 'health-group-vals');
    for (const m of g.values) {
      const chip = makeEl('span', 'health-group-val');
      chip.append(makeEl('span', 'health-group-val-label', m.label + ' '),
        makeEl('span', bandClass(m.band), fmtVal(m.value, m.unit)));
      vals.appendChild(chip);
    }
    row.appendChild(vals);
    wrap.appendChild(row);
  }
  return wrap;
}

// ── Snapshot timeline (shared by summary + deep dive) ────────────────────────

function renderSnapshotTimeline(ctx) {
  const wrap = makeEl('div', 'health-timeline');
  const selStep = ctx.currentSnapshot()?.step;
  ctx.state.history.forEach((hsnap) => {
    const verdict = hsnap.diag.health?.overall || hsnap.status || 'unknown';
    const cls = verdict === 'ok' ? 'diag-ok' : verdict === 'warn' ? 'diag-warn' : verdict === 'error' ? 'diag-error' : '';
    const chip = makeEl('button', `health-chip ${cls}` + (hsnap.step === selStep ? ' selected' : ''), String(hsnap.step));
    chip.title = `step ${hsnap.step} · ${verdict}${hsnap.created_at ? ' · ' + hsnap.created_at : ''}`;
    chip.addEventListener('click', () => { ctx.state.view.selectedStep = hsnap.step; ctx.renderHealthTab(); });
    wrap.appendChild(chip);
  });
  const latest = makeEl('button', 'btn-ghost health-latest', 'Latest');
  latest.addEventListener('click', () => { ctx.state.view.selectedStep = null; ctx.renderHealthTab(); });
  wrap.appendChild(latest);
  return wrap;
}
