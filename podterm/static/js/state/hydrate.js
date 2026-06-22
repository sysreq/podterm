import { fetchJson } from '../api.js';
import { emit } from './bus.js';
import { ingestMetric } from './metrics.js';
import { getPodState } from './pods.js';

// Hydration tracks an explicit lifecycle ('idle' | 'loading' | 'loaded' |
// 'failed') so a failed load can be retried but a successful one is never
// refetched. The authoritative "already loaded" gate stays the legacy boolean
// (`state.hydrated` / `state.runRowHydrated` from freshState(), which pods.js's
// resetRunState() also drives) — we only flip it to `true` AFTER a successful
// fetch+parse, never before, and never in the failure branch. The in-flight
// guard (`state.hydrating` / `state.runRowHydrating`) dedupes concurrent calls.
export async function hydrateFromDb(podId, force = false) {
  const state = getPodState(podId);
  if (state.hydrated && !force) return;
  if (state.hydrating) {
    // A fetch is already in flight. A normal call is just deduped, but a forced
    // gap-fill (e.g. stream.js reopen after a disconnect) must not be silently
    // dropped — the in-flight fetch may have started before the gap. Defer it
    // so it re-runs once the current fetch settles.
    if (force) state.rehydratePending = true;
    return;
  }
  state.hydrating = true;
  state.hydrateStatus = 'loading';

  try {
    let rows;
    try {
      rows = await fetchJson(`/api/runs/${encodeURIComponent(podId)}/metrics`);
    } catch {
      state.hydrateStatus = 'failed'; // leave `hydrated` false so a later call retries
      return;
    }
    if (!Array.isArray(rows)) {
      state.hydrateStatus = 'failed';
      return;
    }

    // An empty array is a successful (if vacuous) load — the run simply has no
    // metrics yet. Mark it loaded so we don't retry, but skip the merge/emit.
    if (rows.length) {
      for (const row of rows) ingestMetric(state, row);
      state.metricHistory.sort((a, b) => a.step - b.step);
      state.evals.sort((a, b) => a.step - b.step);
      state.lastMetric = state.metricHistory[state.metricHistory.length - 1];
    }
    state.hydrated = true;
    state.hydrateStatus = 'loaded';
    if (rows.length) emit('pod:hydrated', { podId });
  } finally {
    state.hydrating = false;
    if (state.rehydratePending) {
      state.rehydratePending = false;
      hydrateFromDb(podId, true); // run the deferred forced gap-fill
    }
  }
}

export async function hydrateRunRow(podId, force = false) {
  const state = getPodState(podId);
  if ((state.runRowHydrated && !force) || state.runRowHydrating) return state.runRow;
  state.runRowHydrating = true;
  state.runRowStatus = 'loading';

  try {
    state.runRow = await fetchJson(`/api/runs/${encodeURIComponent(podId)}`);
    state.runRowHydrated = true;
    state.runRowStatus = 'loaded';
    emit('pod:run-row', { podId, runRow: state.runRow });
  } catch {
    // Leave `runRowHydrated` false so a transient failure retries on the next call.
    state.runRowStatus = 'failed';
  } finally {
    state.runRowHydrating = false;
  }
  return state.runRow;
}
