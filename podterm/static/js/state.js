// Shared app state, per-pod live state, SSE ingestion, and the pub/sub bus.
// SSE handlers only mutate buffers and emit bus events — no DOM access here.
import { fetchJson } from './api.js';

export const app = {
  activePod: null,
  pods: [],
  historyRuns: [],
  selectedRuns: new Set(),
  variantLookup: {},
};

// ── Pub/sub bus ──
const bus = new EventTarget();
export function emit(type, detail = {}) {
  bus.dispatchEvent(new CustomEvent(type, { detail }));
}
export function on(type, fn) {
  bus.addEventListener(type, (e) => fn(e.detail));
}

// ── Per-pod state ──
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
    summary: null,     // {final_val_bpb, best_val_bpb, final_val_loss}
    diagnostic: null,  // latest off-pod health: {step, status, health} (from SSE or the panel's fetch)
    phase: null,
    finished: false,
    exitCode: null,
    lastMetric: null,
    runRow: null, // /api/runs row for this pod (started_at, cost_per_hr, …)
    // Baseline comparison
    baselineRunId: null,
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

// ── Log classification (level stored once, at ingest) ──
const RE_STEP_LINE = /^step:(\d+)\//;
const RE_ERROR = /\b(error|traceback|exception|fatal|failed)\b/i;
const RE_WARN = /\bwarn(ing)?\b/i;

function classifyLog(text) {
  const stepMatch = text.match(RE_STEP_LINE);
  if (stepMatch) return { level: 'metric', step: Number(stepMatch[1]) };
  if (text.startsWith('==>')) return { level: 'metric', step: null };
  if (RE_ERROR.test(text)) return { level: 'error', step: null };
  if (RE_WARN.test(text)) return { level: 'warn', step: null };
  return { level: 'info', step: null };
}

const LOG_CAP = 1000;
function pushLog(state, text) {
  const { level, step } = classifyLog(text);
  state.logLines.push({ text, level, step });
  if (state.logLines.length > LOG_CAP) state.logLines.splice(0, state.logLines.length - 800);
}

// ── Metric ingestion ──
// Val-only metric events carry train_loss: 0.0 — never let them clobber a real loss.
function mergeMetric(target, m) {
  for (const [k, v] of Object.entries(m)) {
    if (v == null) continue;
    if (k === 'train_loss' && !(v > 0)) continue;
    target[k] = v;
  }
}

function ingestMetric(state, m) {
  const existing = state.metricByStep.get(m.step);
  let merged;
  let isNew = false;
  if (existing) {
    mergeMetric(existing, m);
    merged = existing;
  } else {
    merged = { step: m.step };
    mergeMetric(merged, m);
    state.metricByStep.set(m.step, merged);
    state.metricHistory.push(merged);
    isNew = true;
  }
  if (m.val_bpb != null && !state.evalSteps.has(m.step)) {
    state.evalSteps.add(m.step);
    state.evals.push({ step: m.step, val_bpb: m.val_bpb, val_loss: m.val_loss ?? null });
  }
  if (!state.lastMetric || merged.step >= state.lastMetric.step) state.lastMetric = merged;
  return { merged, isNew };
}

// New run started on the same pod — clear run-scoped buffers (keep logs and pod info).
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
  emit('pod:reset', { podId });
}

// ── SSE stream ──
export function ensurePodStream(podId) {
  const state = getPodState(podId);
  if (state.es) return;

  const es = new EventSource(`/api/stream/${podId}`);
  state.es = es;

  es.addEventListener('log', (e) => {
    const d = JSON.parse(e.data);
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

  es.addEventListener('diagnostic', (e) => {
    // Off-pod model-health result for one snapshot. Stash the verdict for the live card; the panel
    // refetches the full series for its drill-down.
    const d = JSON.parse(e.data);
    state.diagnostic = { step: d.step, status: d.status, health: d.health || null };
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

// ── DB hydration ──
// The SSE stream has no replay for late subscribers; seed history from SQLite
// (flushed every few seconds server-side) so a refresh mid-run isn't blank.
export async function hydrateFromDb(podId, force = false) {
  const state = getPodState(podId);
  if (state.hydrated && !force) return;
  state.hydrated = true; // set first so concurrent calls don't double-fetch

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
