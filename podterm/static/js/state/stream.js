import { emit } from './bus.js';
import { hydrateFromDb } from './hydrate.js';
import { pushLog } from './logs.js';
import { ingestMetric } from './metrics.js';
import { getPodState, resetRunState } from './pods.js';

export function ensurePodStream(podId) {
  const state = getPodState(podId);
  if (state.es) return;

  const es = new EventSource(`/api/stream/${podId}`);
  state.es = es;

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
