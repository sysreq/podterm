// Pure derivation math for every computed number on the dashboard:
// ETA, cost, required pace, ahead/behind, deltas. No DOM, no fetch, no
// Date.now() — callers pass timestamps in. Unit tests: derive.test.mjs
// (node --test podterm/static/js/derive.test.mjs).

// Spec-mandated EMA windows live here, not in views.
export const ETA_EMA_N = 50;   // Projected Finish uses EMA-50 of ms/step
export const AVG_EMA_N = 100;  // Avg ms/step card uses EMA-100

export function ema(values, n) {
  if (!values || !values.length) return null;
  const tail = values.slice(-n);
  const alpha = 2 / (tail.length + 1);
  let e = tail[0];
  for (let i = 1; i < tail.length; i++) e = alpha * tail[i] + (1 - alpha) * e;
  return e;
}

// ── Time / cost ──
export function etaMs(remainingSteps, msPerStep) {
  if (remainingSteps == null || msPerStep == null || remainingSteps < 0) return null;
  return remainingSteps * msPerStep;
}

export function costSoFar(elapsedWallMs, costPerHr) {
  if (elapsedWallMs == null || costPerHr == null) return null;
  return (elapsedWallMs / 3_600_000) * costPerHr;
}

export function projectedTotalCost(elapsedWallMs, eta, costPerHr) {
  if (elapsedWallMs == null || eta == null || costPerHr == null) return null;
  return ((elapsedWallMs + eta) / 3_600_000) * costPerHr;
}

export function elapsedWallMs(startedAtIso, nowMs) {
  if (!startedAtIso || nowMs == null) return null;
  const t = Date.parse(startedAtIso);
  if (Number.isNaN(t)) return null;
  return Math.max(0, nowMs - t);
}

// ── Baseline race math ──
// Last baseline sample at or before `step` (metrics are sparse — every ~250 steps).
export function baselineAtStep(baselineByStep, sortedSteps, step) {
  if (!sortedSteps || !sortedSteps.length || step == null) return null;
  let lo = 0, hi = sortedSteps.length - 1, found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (sortedSteps[mid] <= step) { found = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return found >= 0 ? baselineByStep[sortedSteps[found]] : null;
}

// Instantaneous + cumulative deltas vs baseline. Negative = ahead.
export function paceDelta(metric, baselineSample) {
  if (!metric || !baselineSample) return null;
  if (!metric.step_avg_ms || !baselineSample.step_avg_ms) return null;
  if (!metric.train_time_ms || !baselineSample.train_time_ms) return null;
  return {
    perStepMs: metric.step_avg_ms - baselineSample.step_avg_ms,
    cumulativeMs: metric.train_time_ms - baselineSample.train_time_ms,
  };
}

// Pace needed from here on to finish in the baseline's total time.
export function requiredPaceMs(baselineTotalTimeMs, currentElapsedTrainMs, remainingSteps) {
  if (baselineTotalTimeMs == null || currentElapsedTrainMs == null) return null;
  if (remainingSteps == null || remainingSteps <= 0) return null;
  return (baselineTotalTimeMs - currentElapsedTrainMs) / remainingSteps;
}

// Margin at the finish line if the current pace holds. Positive = finish ahead.
// Distinct from the current cumulative lead — do not conflate the two.
export function projectedFinishMarginMs(requiredMs, currentEmaMs, remainingSteps) {
  if (requiredMs == null || currentEmaMs == null) return null;
  if (remainingSteps == null || remainingSteps <= 0) return null;
  return (requiredMs - currentEmaMs) * remainingSteps;
}

// Single source for the hero card, race banner, and sidebar pace lines.
export function raceStatus({ metric, baselineByStep, baselineSteps, baselineTotalTimeMs, emaMs }) {
  if (!metric) return { state: 'no-data' };
  const empty = {
    state: 'no-baseline',
    perStepMs: null, cumulativeMs: null, requiredMs: null, projectedMarginMs: null,
  };
  if (!baselineSteps || !baselineSteps.length) return empty;
  const sample = baselineAtStep(baselineByStep, baselineSteps, metric.step);
  const pace = paceDelta(metric, sample);
  if (!pace) return empty;
  const remaining = metric.total_steps != null && metric.step != null
    ? metric.total_steps - metric.step : null;
  const requiredMs = requiredPaceMs(baselineTotalTimeMs, metric.train_time_ms, remaining);
  const projectedMarginMs = projectedFinishMarginMs(requiredMs, emaMs, remaining);
  return {
    state: pace.cumulativeMs <= 0 ? 'ahead' : 'behind',
    perStepMs: pace.perStepMs,
    cumulativeMs: pace.cumulativeMs,
    requiredMs,
    projectedMarginMs,
  };
}

// ── Deltas ──
// Change vs the sample nearest to `stepsAgo` steps back. Sparsity-tolerant:
// metrics arrive every ~250 steps, so this finds real samples by step search.
export function deltaVsStepsAgo(history, key, currentStep, stepsAgo = 100) {
  if (!history || !history.length || currentStep == null) return null;
  let cur = null, past = null;
  for (let i = history.length - 1; i >= 0; i--) {
    const m = history[i];
    const v = m[key];
    if (v == null || (key === 'train_loss' && !(v > 0))) continue;
    if (cur == null) {
      if (m.step <= currentStep) cur = { step: m.step, value: v };
      continue;
    }
    if (m.step <= cur.step - stepsAgo) { past = { step: m.step, value: v }; break; }
  }
  if (!cur || !past) return null;
  return { current: cur.value, delta: cur.value - past.value, stepsSpanned: cur.step - past.step };
}

export function evalDelta(evals) {
  if (!evals || !evals.length) return null;
  const cur = evals[evals.length - 1];
  const prev = evals.length > 1 ? evals[evals.length - 2] : null;
  return { current: cur.val_bpb, step: cur.step, delta: prev ? cur.val_bpb - prev.val_bpb : null };
}

// ── Metadata parsing ──
// "NVIDIA H100 80GB HBM3" -> 80. Null when the name carries no memory size.
export function parseGpuMemGiB(gpuTypeName) {
  if (!gpuTypeName) return null;
  const m = String(gpuTypeName).match(/(\d+)\s*GB/i);
  return m ? Number(m[1]) : null;
}
