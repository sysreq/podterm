import { emit } from './bus.js';
import { hydrateFromDb } from './hydrate.js';
import { pushLog } from './logs.js';
import { ingestMetric } from './metrics.js';
import { getPodState, resetRunState } from './pods.js';

// Browsers cap ~6 SSE connections per origin on HTTP/1.1, so we stream at most
// MAX_STREAMS pods at once. The reconciler below always reserves a slot for the
// active pod and LRU-evicts the least-recently-active background streams.
const MAX_STREAMS = 4;

// Monotonic recency clock: higher = more recently touched. reconcileStreams uses
// it to decide which background streams to keep when over the cap.
let recencyClock = 0;
const recency = new Map(); // podId -> recency tick

function touch(podId) {
  recency.set(podId, ++recencyClock);
}

// Pods this manager has an open EventSource for, keyed to the actual EventSource
// so we can close/identify streams without reaching back into pod state (which
// getPodState() would lazily resurrect for torn-down pods).
const openStreams = new Map(); // podId -> EventSource

function closePodStream(podId) {
  const es = openStreams.get(podId);
  if (es) es.close();
  openStreams.delete(podId);
  recency.delete(podId); // a closed stream no longer competes for a slot
  // Clear the app-visible handle too, but only if it still points at our stream
  // (dropPodState may already have replaced/removed the pod state).
  const state = getPodState(podId);
  if (state.es === es) state.es = null;
}

export function ensurePodStream(podId) {
  const state = getPodState(podId);
  touch(podId); // any (re)connect request marks the pod recently active
  if (state.es) return;

  const es = new EventSource('/api/stream/' + encodeURIComponent(podId));
  state.es = es;
  openStreams.set(podId, es);

  es.addEventListener('log', (e) => {
    const d = JSON.parse(e.data);
    if (d.line.startsWith('Event daemon connected')) state.containerConnected = true;
    if (d.line.startsWith('No event-daemon token') || d.line.startsWith('Event daemon auth failed')) {
      state.streamUnavailable = true;
    }
    pushLog(state, d.line);
    emit('pod:log', { podId, line: d.line });
  });

  es.addEventListener('metric', (e) => {
    const m = JSON.parse(e.data);
    const { merged, isNew } = ingestMetric(state, m);
    emit('pod:metric', {
      podId,
      m: merged,
      isNew,
      trainPoint: isNew && merged.train_loss > 0,
    });
  });

  es.addEventListener('memory', (e) => {
    state.memory = JSON.parse(e.data);
    emit('pod:memory', { podId, memory: state.memory });
  });

  es.addEventListener('telemetry', (e) => {
    const t = JSON.parse(e.data);
    state.telemetry = t;
    if (t.gpu_util_pct != null) {
      state.telemetryHistory.push(t.gpu_util_pct);
      if (state.telemetryHistory.length > 120) state.telemetryHistory.splice(0, 20);
    }
    emit('pod:telemetry', { podId, telemetry: t });
  });

  es.addEventListener('pull', (e) => {
    state.boot = JSON.parse(e.data);
    emit('pod:boot', { podId });
  });

  es.addEventListener('info', (e) => {
    Object.assign(state.info, JSON.parse(e.data));
    emit('pod:info', { podId, info: state.info });
  });

  es.addEventListener('summary', (e) => {
    state.summary = JSON.parse(e.data);
    emit('pod:summary', { podId, summary: state.summary });
  });

  es.addEventListener('snapshot', (e) => {
    // The pipeline emits this when a snapshot lands, before off-pod diagnostics run. Mark PENDING
    // until the matching 'diagnostic' arrives (cleared below).
    const d = JSON.parse(e.data);
    if (d.step != null) state.pendingSnapshotStep = d.step;
    emit('pod:snapshot', { podId, step: d.step });
  });

  es.addEventListener('diagnostic', (e) => {
    // Off-pod model-health result for one snapshot. Stash the verdict for the live card; the panel
    // refetches the full series for its drill-down.
    const d = JSON.parse(e.data);
    state.diagnostic = { step: d.step, status: d.status, health: d.health || null };
    if (state.pendingSnapshotStep != null && d.step != null && d.step >= state.pendingSnapshotStep) {
      state.pendingSnapshotStep = null;
    }
    emit('pod:diagnostic', { podId, diag: d });
  });

  es.addEventListener('phase', (e) => {
    const d = JSON.parse(e.data);
    state.phase = d.phase || null;
    if (d.phase && d.phase.includes('Starting Training')) resetRunState(podId);
    if (d.phase && d.phase.includes('Training finished')) {
      state.finished = true;
      state.exitCode = d.exit_code ?? null;
    }
    emit('pod:phase', { podId, phase: d.phase, exitCode: d.exit_code ?? null });
  });

  // EventSource auto-reconnects; refill any gap from the DB on reopen.
  es.onerror = () => {
    state.needsRehydrate = true;
  };
  es.onopen = () => {
    if (state.needsRehydrate) {
      state.needsRehydrate = false;
      hydrateFromDb(podId, true);
    }
  };
}

// Bump a pod's recency without (re)opening its stream — call this when a pod is
// selected so the reconciler protects it from LRU eviction next cycle.
export function markPodActive(podId) {
  touch(podId);
}

// Reconcile desired vs. actual EventSource streams. Called on every pod refresh
// and on selection change. Closes streams for pods that are gone, always reserves
// a slot for the active pod (so selecting a 5th pod evicts a background stream
// rather than dropping the selection), and LRU-evicts the least-recently-active
// background streams beyond MAX_STREAMS. Returns the pod ids whose streams were
// freshly (re)opened so the caller can hydrate them from the DB.
export function reconcileStreams(pods, activePod) {
  // Self-heal the mirror: dropPodState() (pod teardown) closes state.es out of
  // band, so an entry here may point at a stream the app already discarded. Drop
  // any whose pod state no longer references our EventSource. Compare identity
  // rather than truthiness so a freshly-resurrected blank state doesn't mask it.
  for (const [podId, es] of [...openStreams]) {
    if (getPodState(podId).es !== es) {
      openStreams.delete(podId);
      recency.delete(podId);
    }
  }

  const runningIds = new Set(pods.filter((p) => p.desiredStatus === 'RUNNING').map((p) => p.id));
  // The active pod is always eligible for a stream even before it flips to
  // RUNNING (queued/booting) or after it exits, so a selected run keeps its live
  // feed; everything else must be RUNNING to qualify.
  const eligible = new Set(runningIds);
  if (activePod) eligible.add(activePod);

  // 1. Close streams for pods that are no longer eligible (gone / not running).
  for (const podId of [...openStreams.keys()]) {
    if (!eligible.has(podId)) closePodStream(podId);
  }

  // 2. Build the desired set, capped at MAX_STREAMS. The active pod always gets
  //    the first slot; the rest are filled by most-recently-active.
  const candidates = [...eligible].sort((a, b) => {
    if (a === activePod) return -1;
    if (b === activePod) return 1;
    return (recency.get(b) || 0) - (recency.get(a) || 0); // newer first
  });
  const desired = new Set(candidates.slice(0, MAX_STREAMS));

  // 3. Evict open streams that didn't make the cut (least-recently-active first).
  for (const podId of [...openStreams.keys()]) {
    if (!desired.has(podId)) closePodStream(podId);
  }

  // 4. Open any desired streams that aren't open yet; report them for hydration.
  const opened = [];
  for (const podId of desired) {
    if (!openStreams.has(podId)) {
      ensurePodStream(podId);
      opened.push(podId);
    }
  }
  return opened;
}
