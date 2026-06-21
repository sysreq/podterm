import { emit } from './bus.js';

const podStates = {};

function freshState() {
  return {
    es: null,
    hydrated: false,
    needsRehydrate: false,
    // Metrics, merged per step (val events at a step fold into the train entry).
    metricByStep: new Map(),
    metricHistory: [], // ordered by step
    evals: [],         // {step, val_bpb, val_loss}
    evalSteps: new Set(),
    logLines: [],      // {text, level, step|null}
    info: {},          // gpu_type, gpu_count, driver_version, cuda_version, seed, seq_len, batch_tokens, commit_hash, commit_msg, model_params
    memory: null,      // {peak_mib, reserved_mib}
    telemetry: null,   // {cpu_pct, mem_pct, gpu_util_pct, gpu_mem_pct, uptime_s}
    telemetryHistory: [], // recent gpu_util_pct samples
    boot: null,        // {stage, image, message, layers, total, complete, pct, done}
    containerConnected: false,
    streamUnavailable: false,
    summary: null,     // {final_val_bpb, best_val_bpb, final_val_loss}
    diagnostic: null,  // latest off-pod health: {step, status, health} (from SSE or the panel's fetch)
    pendingSnapshotStep: null, // a 'snapshot' SSE arrived; diagnostics for it haven't landed yet (PENDING)
    phase: null,
    finished: false,
    exitCode: null,
    lastMetric: null,
    runRow: null, // /api/runs row for this pod (started_at, cost_per_hr, …)
    runRowHydrated: false,
    runRowHydrating: false,
    // Baseline comparison
    baselineRunId: null,
    baselineRunRow: null,        // /api/runs row for the selected baseline (for its finish budget)
    baselineFinishBudgetMs: null, // baseline's configured max-wallclock horizon, ms
    baselineByStep: {},
    baselineSteps: [],
    baselineTotalTimeMs: null,
    baselineX: [],
    baselineY: [],
    baselineRaw: [],
  };
}


export function getPodState(podId) {
  if (!podStates[podId]) podStates[podId] = freshState();
  return podStates[podId];
}

export function dropPodState(podId) {
  const s = podStates[podId];
  if (s && s.es) s.es.close();
  delete podStates[podId];
}

export function resetRunState(podId) {
  const s = getPodState(podId);
  s.metricByStep.clear();
  s.metricHistory.length = 0;
  s.evals.length = 0;
  s.evalSteps.clear();
  s.lastMetric = null;
  s.summary = null;
  s.finished = false;
  s.exitCode = null;
  s.pendingSnapshotStep = null;
  s.runRowHydrated = false;
  emit('pod:reset', { podId });
}
