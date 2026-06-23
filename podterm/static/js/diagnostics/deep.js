import { getPodState } from '../state.js';
import { escapeHtml } from '../dom.js';
import { latestSnapshot, referenceDoc } from './selectors.js';
import { bandClass, fmtRowMetrics, makeEl, sectionStatusLabel, sevRank } from './format.js';

// How long the "updated" tag lingers on an expanded section after a new snapshot
// lands (matched by the CSS fade-out animation duration).
const UPDATED_TAG_MS = 10_000;

const el = (id) => document.getElementById(id);

export function renderHealthDeepDive(ctx) {
  const wrap = makeEl('div', 'health-deep');
  wrap.appendChild(renderDeepToolbar(ctx));
  const snap = ctx.currentSnapshot();
  const refDoc = referenceDoc(ctx.state.history, snap.step, ctx.state.view.diffMode, ctx.state.baselineHistory);

  let sections = [...(snap.diag.sections || [])].sort((a, b) => sevRank(b.status) - sevRank(a.status));
  if (ctx.state.view.onlyProblems) sections = sections.filter((s) => ['error', 'partial', 'warn'].includes(s.status));
  else if (!ctx.state.view.includeUnavailable) sections = sections.filter((s) => s.status !== 'skipped');
  if (ctx.state.view.filterText) {
    const q = ctx.state.view.filterText.toLowerCase();
    sections = sections.filter((s) => s.name.toLowerCase().includes(q)
      || Object.keys(s.rows || {}).some((k) => k.toLowerCase().includes(q)));
  }

  const layout = makeEl('div', 'health-deep-layout');
  layout.append(renderSectionRail(sections), renderDetailMain(ctx, sections, refDoc));
  wrap.appendChild(layout);
  wrap.appendChild(renderRawJson(snap));
  return wrap;
}

function renderDeepToolbar(ctx) {
  const bar = makeEl('div', 'health-toolbar');
  const back = makeEl('button', 'btn-ghost', '← Back to Summary');
  back.addEventListener('click', () => { ctx.state.view.mode = 'summary'; ctx.renderHealthTab(); });

  // Snapshot selector.
  const sel = makeEl('select', 'health-snap-select');
  const curStep = ctx.currentSnapshot()?.step;
  [...ctx.state.history].reverse().forEach((h) => {
    const o = makeEl('option', null, `step ${h.step} · ${h.diag.health?.overall || h.status || '?'}`);
    o.value = String(h.step);
    if (h.step === curStep) o.selected = true;
    sel.appendChild(o);
  });
  sel.addEventListener('change', () => {
    const latest = latestSnapshot(ctx.state.history);
    ctx.state.view.selectedStep = latest && Number(sel.value) === latest.step ? null : Number(sel.value);
    ctx.renderHealthTab();
  });

  // Diff-mode buttons.
  const diffWrap = makeEl('div', 'health-diff-modes');
  const hasBaseline = Boolean(ctx.state.activeRunId && getPodState(ctx.state.activeRunId).baselineRunId);
  const MODES = [['previous', 'Diff: Previous'], ['baseline', 'Diff: Baseline'], ['absolute', 'Absolute']];
  for (const [mode, label] of MODES) {
    const b = makeEl('button', 'health-diff-btn' + (ctx.state.view.diffMode === mode ? ' active' : ''), label);
    if (mode === 'baseline' && !hasBaseline) { b.disabled = true; b.title = 'Select a baseline on the Live tab first'; }
    b.addEventListener('click', () => {
      ctx.state.view.diffMode = mode;
      if (mode === 'baseline') ctx.ensureBaseline();
      ctx.renderHealthTab();
    });
    diffWrap.appendChild(b);
  }

  const search = makeEl('input', 'health-search');
  search.placeholder = 'Filter sections…';
  search.value = ctx.state.view.filterText;
  search.addEventListener('input', () => { ctx.state.view.filterText = search.value; ctx.renderHealthTab(); });

  const toggles = makeEl('div', 'health-toggles');
  toggles.append(
    checkbox('Only problems', ctx.state.view.onlyProblems, (v) => { ctx.state.view.onlyProblems = v; ctx.renderHealthTab(); }),
    checkbox('Include unavailable', ctx.state.view.includeUnavailable, (v) => { ctx.state.view.includeUnavailable = v; ctx.renderHealthTab(); }),
  );

  bar.append(back, sel, diffWrap, search, toggles);
  return bar;
}

function checkbox(label, checked, onChange) {
  const wrap = makeEl('label', 'health-check');
  const box = document.createElement('input');
  box.type = 'checkbox';
  box.checked = checked;
  box.addEventListener('change', () => onChange(box.checked));
  wrap.append(box, document.createTextNode(' ' + label));
  return wrap;
}

function renderSectionRail(sections) {
  const rail = makeEl('div', 'health-section-rail');
  if (!sections.length) { rail.appendChild(makeEl('div', 'health-empty-msg', 'No sections match.')); return rail; }
  sections.forEach((s, i) => {
    const item = makeEl('button', 'health-rail-item');
    item.append(makeEl('span', 'health-rail-name', s.name),
      makeEl('span', `health-badge ${bandClass(s.status === 'ok' ? 'good' : s.status)}`, sectionStatusLabel(s.status)));
    item.addEventListener('click', () => {
      const target = el(`health-sec-${i}`);
      if (target) { target.open = true; target.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
    });
    rail.appendChild(item);
  });
  return rail;
}

function renderDetailMain(ctx, sections, refDoc) {
  const main = makeEl('div', 'health-detail-main');
  if (refDoc === null && ctx.state.view.diffMode !== 'absolute') {
    main.appendChild(makeEl('div', 'health-empty-msg',
      ctx.state.view.diffMode === 'baseline' ? 'No baseline diagnostics to compare against.'
        : 'No previous snapshot to compare against.'));
  }
  const refSection = (name) => (refDoc?.sections || []).find((s) => s.name === name);
  // Open-state is keyed by section name and persisted in view.openSections so it
  // survives re-renders (diff-mode switch, new snapshot, snapshot selection). On
  // first render it's seeded from the severity default (problem sections expanded).
  if (ctx.state.view.openSections == null) {
    ctx.state.view.openSections = new Set(
      sections.filter((s) => ['error', 'partial', 'warn'].includes(s.status)).map((s) => s.name));
  }
  const openSet = ctx.state.view.openSections;
  // A fresh snapshot that changed the live view flags expanded sections as updated.
  const viewingLatest = ctx.state.view.selectedStep == null;
  const updatedFresh = ctx.state.lastUpdateTs > 0 && (Date.now() - ctx.state.lastUpdateTs) < UPDATED_TAG_MS;
  sections.forEach((s, i) => {
    const sec = makeEl('details', 'health-section');
    sec.id = `health-sec-${i}`;
    sec.open = openSet.has(s.name);
    sec.addEventListener('toggle', () => { if (sec.open) openSet.add(s.name); else openSet.delete(s.name); });
    const summary = document.createElement('summary');
    summary.innerHTML = `<span class="diag-sec-name">${escapeHtml(s.name)}</span>`
      + `<span class="diag-sec-status diag-${escapeHtml(s.status)}">${escapeHtml(sectionStatusLabel(s.status))}`
      + `${s.reason ? ': ' + escapeHtml(s.reason) : ''}</span>`;
    sec.appendChild(summary);
    if (sec.open && updatedFresh && viewingLatest) sec.appendChild(makeEl('div', 'diag-sec-updated', 'updated'));

    const rows = s.rows || {};
    const refRows = refSection(s.name)?.rows || {};
    if (Object.keys(rows).length) {
      const table = makeEl('table', 'health-metric-table');
      for (const [rk, metrics] of Object.entries(rows)) {
        const tr = document.createElement('tr');
        const td0 = makeEl('td', 'diag-row-key', rk);
        const td1 = document.createElement('td');
        td1.innerHTML = fmtRowMetrics(metrics, ctx.state.view.diffMode === 'absolute' ? null : refRows[rk]);
        tr.append(td0, td1);
        table.appendChild(tr);
      }
      sec.appendChild(table);
    } else {
      // The probe ran ('complete') but emitted no metric rows — e.g. this model has
      // no learnable params for the check. Say so instead of expanding to a blank.
      const notes = Array.isArray(s.notes) ? s.notes.filter(Boolean) : [];
      sec.appendChild(makeEl('div', 'health-section-empty',
        notes.length ? notes.join(' ') : 'No metrics for this section on this model.'));
    }
    main.appendChild(sec);
  });
  return main;
}

function renderRawJson(snapshot) {
  const raw = makeEl('details', 'diag-raw');
  raw.appendChild(makeEl('summary', null, 'Raw JSON'));
  const pre = document.createElement('pre');
  pre.textContent = JSON.stringify(snapshot.diag, null, 2);
  raw.appendChild(pre);
  return raw;
}
