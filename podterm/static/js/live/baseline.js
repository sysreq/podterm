import { app, getPodState } from '../state.js';
import * as derive from '../derive.js';
import { scaleY, updateBaselineTrace } from '../charts.js';
import { renderConfigPanel } from '../configpanel.js';
import { fetchJson } from '../api.js';
import { updateKpis } from './kpis.js';

const PINNED_BASELINE_KEY = 'podterm:pinnedBaselineRunId';

function pinnedBaselineId() {
  try {
    return localStorage.getItem(PINNED_BASELINE_KEY);
  } catch {
    return null;
  }
}

function setPinnedBaselineId(runId) {
  try {
    if (runId) localStorage.setItem(PINNED_BASELINE_KEY, runId);
    else localStorage.removeItem(PINNED_BASELINE_KEY);
  } catch {}
}

function optionExists(select, value) {
  return !!value && Array.from(select.options).some((opt) => opt.value === value);
}

function resetBaselineState(state, runId) {
  state.baselineRunId = runId || null;
  state.baselineRunRow = null;
  state.baselineFinishBudgetMs = null;
  state.baselineByStep = {};
  state.baselineSteps = [];
  state.baselineTotalTimeMs = null;
  state.baselineX = [];
  state.baselineY = [];
  state.baselineRaw = [];
}

function observedTrainingTimeMs(metrics) {
  for (let i = metrics.length - 1; i >= 0; i--) {
    const m = metrics[i];
    if (m && m.train_loss > 0 && m.train_time_ms > 0) return m.train_time_ms;
  }
  return null;
}

function updatePinButton() {
  const btn = document.getElementById('baseline-pin');
  const sel = document.getElementById('baseline-select');
  if (!btn || !sel) return;
  const selected = sel.value || null;
  const pinned = pinnedBaselineId();
  const active = !!selected && selected === pinned;
  btn.disabled = !selected && !pinned;
  btn.classList.toggle('active', active);
  btn.setAttribute('aria-pressed', active ? 'true' : 'false');
  btn.textContent = active ? 'Pinned' : (!selected && pinned) ? 'Clear Pin' : 'Pin';
  btn.title = selected
    ? active ? 'Unpin this baseline' : 'Pin selected baseline for future runs'
    : pinned ? 'Clear pinned baseline' : 'Select a baseline to pin';
}

export async function loadBaselineOptions(podId) {
  const sel = document.getElementById('baseline-select');
  sel.innerHTML = '<option value="">None</option>';
  try {
    const runs = await fetchJson('/api/runs?limit=30');
    if (app.activePod !== podId) return;
    const state = getPodState(podId);
    state.runRow = runs.find((r) => r.run_id === podId) || state.runRow;
    for (const r of runs) {
      if (r.run_id === podId) continue;
      if (!r.total_steps) continue;
      const bpb = r.best_val_bpb != null ? ` | BPB ${r.best_val_bpb.toFixed(4)}` : '';
      const label = `${r.pod_name || r.run_id} (${r.branch || '?'}${bpb})`;
      const opt = document.createElement('option');
      opt.value = r.run_id;
      opt.textContent = label;
      sel.appendChild(opt);
    }
    updateKpis(podId);
    renderConfigPanel(podId);
  } catch {}
  const state = getPodState(podId);
  const pinned = pinnedBaselineId();
  const selected = optionExists(sel, state.baselineRunId)
    ? state.baselineRunId
    : (pinned !== podId && optionExists(sel, pinned)) ? pinned : null;
  sel.value = selected || '';
  updatePinButton();
  if (selected && state.baselineRunId !== selected) {
    await applyBaselineSelection(podId, selected);
  }
}

export async function onBaselineChange() {
  if (!app.activePod) return;
  await applyBaselineSelection(app.activePod, document.getElementById('baseline-select').value);
}

export async function onPinBaselineClick() {
  const sel = document.getElementById('baseline-select');
  const selected = sel.value || null;
  const pinned = pinnedBaselineId();
  if (selected) {
    setPinnedBaselineId(selected === pinned ? null : selected);
  } else if (pinned) {
    setPinnedBaselineId(null);
  }
  updatePinButton();
}

async function applyBaselineSelection(podId, runId) {
  const state = getPodState(podId);
  resetBaselineState(state, runId);
  updatePinButton();

  if (!runId) {
    updateBaselineTrace(state);
    updateKpis(podId);
    return;
  }

  let metrics;
  try {
    metrics = await fetchJson(`/api/runs/${runId}/metrics`);
  } catch {
    updateBaselineTrace(state);
    updateKpis(podId);
    return;
  }
  if (app.activePod !== podId) return;

  for (const m of metrics) {
    state.baselineByStep[m.step] = m;
  }
  state.baselineSteps = metrics.map((m) => m.step).sort((a, b) => a - b);
  state.baselineTotalTimeMs = observedTrainingTimeMs(metrics);

  // The finish-horizon race needs the baseline's configured max-wallclock (its run
  // row), not the last metric timestamp. Fall back to observed total for pre-budget runs.
  try {
    const baselineRow = await fetchJson(`/api/runs/${runId}`);
    if (app.activePod !== podId || state.baselineRunId !== runId) return;
    state.baselineRunRow = baselineRow;
    state.baselineFinishBudgetMs = derive.configFinishBudgetMs(baselineRow) ?? state.baselineTotalTimeMs;
  } catch {
    state.baselineFinishBudgetMs = state.baselineTotalTimeMs;
  }

  const baselineMetrics = metrics.filter((m) => m.train_loss);
  state.baselineX = baselineMetrics.map((m) => m.step);
  state.baselineY = baselineMetrics.map((m) => scaleY(m.train_loss));
  state.baselineRaw = baselineMetrics.map((m) => m.train_loss);

  updateBaselineTrace(state);
  updateKpis(podId);
}
