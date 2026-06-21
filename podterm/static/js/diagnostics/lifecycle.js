import { app, emit, getPodState, on } from '../state.js';
import { fetchJson } from '../api.js';
import { emptyState } from './format.js';
import { latestSnapshot, snapshotByStep } from './selectors.js';
import { renderHealthDeepDive } from './deep.js';
import { renderHealthSummary } from './summary.js';

export const diagState = {
  activeRunId: null,
  history: [],
  baselineHistory: [],
  baselineFetchedFor: null,
  view: {
    selectedStep: null,
    mode: 'summary',
    filterText: '',
    onlyProblems: false,
    includeUnavailable: false,
    diffMode: 'previous',
  },
};

const el = (id) => document.getElementById(id);

export function initDiagnostics() {
  on('pod:hydrated', ({ podId }) => { if (podId === app.activePod) loadDiagnostics(podId); });
  on('pod:diagnostic', ({ podId }) => { if (podId === app.activePod) loadDiagnostics(podId); });
  on('pod:snapshot', ({ podId }) => { if (podId === app.activePod && healthTabVisible()) renderHealthTab(); });
  on('pod:reset', ({ podId }) => {
    if (podId !== diagState.activeRunId) return;
    diagState.history = [];
    diagState.baselineHistory = [];
    diagState.baselineFetchedFor = null;
    diagState.view.selectedStep = null;
    if (healthTabVisible()) renderHealthTab();
  });
}

function healthTabVisible() {
  const t = el('tab-health');
  return Boolean(t && t.classList.contains('active'));
}

export function openHealthTab(podId) {
  if (podId !== diagState.activeRunId) {
    diagState.activeRunId = podId || null;
    diagState.history = [];
    diagState.baselineHistory = [];
    diagState.baselineFetchedFor = null;
    diagState.view.selectedStep = null;
    diagState.view.mode = 'summary';
  }
  renderHealthTab();
  if (podId) loadDiagnostics(podId);
  if (diagState.view.diffMode === 'baseline') ensureBaseline();
}

async function loadDiagnostics(runId) {
  if (!runId) { renderHealthTab(); return; }
  let raw;
  try { raw = await fetchJson(`/api/runs/${runId}/diagnostics`); } catch { return; }
  if (app.activePod !== runId) return;
  const valid = (Array.isArray(raw) ? raw : []).filter((h) => h && h.diag);
  diagState.activeRunId = runId;
  diagState.history = valid;

  const st = getPodState(runId);
  if (valid.length) {
    const l = valid[valid.length - 1];
    st.diagnostic = { step: l.step, status: l.status, health: l.diag.health || null };
    if (st.pendingSnapshotStep != null && l.step >= st.pendingSnapshotStep) {
      st.pendingSnapshotStep = null;
    }
    emit('pod:health', { podId: runId });
  }
  if (healthTabVisible()) renderHealthTab();
}

export async function ensureBaseline() {
  const bid = diagState.activeRunId ? getPodState(diagState.activeRunId).baselineRunId : null;
  if (!bid) {
    diagState.baselineHistory = [];
    diagState.baselineFetchedFor = null;
    return;
  }
  if (diagState.baselineFetchedFor === bid) return;
  diagState.baselineFetchedFor = bid;
  try {
    const raw = await fetchJson(`/api/runs/${bid}/diagnostics`);
    diagState.baselineHistory = (Array.isArray(raw) ? raw : []).filter((h) => h && h.diag);
  } catch {
    diagState.baselineHistory = [];
  }
  if (healthTabVisible()) renderHealthTab();
}

export function currentSnapshot() {
  if (diagState.view.selectedStep != null) {
    return snapshotByStep(diagState.history, diagState.view.selectedStep) || latestSnapshot(diagState.history);
  }
  return latestSnapshot(diagState.history);
}

export function renderHealthTab() {
  const root = el('health-root');
  if (!root) return;
  root.innerHTML = '';
  if (!diagState.activeRunId) {
    root.appendChild(emptyState('No run selected', 'Pick a run from the sidebar to see its model health.'));
    return;
  }
  if (!diagState.history.length) {
    root.appendChild(emptyState('No diagnostics yet',
      'Diagnostics run off-pod on each snapshot. The first verdict appears once a snapshot completes.'));
    return;
  }
  const ctx = { state: diagState, currentSnapshot, renderHealthTab, ensureBaseline, healthTabVisible };
  root.appendChild(diagState.view.mode === 'deep' ? renderHealthDeepDive(ctx) : renderHealthSummary(ctx));
}
