import { fetchJson } from '../api.js';
import { emit } from './bus.js';
import { ingestMetric } from './metrics.js';
import { getPodState } from './pods.js';

export async function hydrateFromDb(podId, force = false) {
  const state = getPodState(podId);
  if (state.hydrated && !force) return;
  state.hydrated = true;

  let rows;
  try {
    rows = await fetchJson(`/api/runs/${podId}/metrics`);
  } catch {
    return;
  }
  if (!Array.isArray(rows) || !rows.length) return;

  for (const row of rows) ingestMetric(state, row);
  state.metricHistory.sort((a, b) => a.step - b.step);
  state.evals.sort((a, b) => a.step - b.step);
  state.lastMetric = state.metricHistory[state.metricHistory.length - 1];
  emit('pod:hydrated', { podId });
}

export async function hydrateRunRow(podId, force = false) {
  const state = getPodState(podId);
  if ((state.runRowHydrated && !force) || state.runRowHydrating) return state.runRow;
  state.runRowHydrating = true;

  try {
    state.runRow = await fetchJson(`/api/runs/${encodeURIComponent(podId)}`);
    state.runRowHydrated = true;
    emit('pod:run-row', { podId, runRow: state.runRow });
  } catch {
    state.runRowHydrated = true;
  } finally {
    state.runRowHydrating = false;
  }
  return state.runRow;
}
