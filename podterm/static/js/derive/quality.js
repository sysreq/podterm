import { ema } from './ema.js';

// ── Quality-vs-time race ──
// The race asks "am I reaching the same loss in less training wall-clock than
// the baseline did?" — not the old "fewer ms to the same step", which read as
// ahead even when the run was converging worse.

export const QUALITY_WINDOW_STEPS = 500; // trailing window for the loss-descent rate
const QUALITY_LOSS_EMA_N = 10;           // smooth the current loss to kill per-step jitter

// Throughput in tokens/sec from the step time and the per-step token budget.
export function throughputTps(msPerStep, batchTokens) {
  if (msPerStep == null || !(msPerStep > 0)) return null;
  if (batchTokens == null || !(batchTokens > 0)) return null;
  return batchTokens / (msPerStep / 1000);
}

// When did the baseline first reach `targetLoss`? Walk samples in step order
// (loss trends down) for the first crossing to <= target, interpolating the
// time linearly in loss between the bracketing samples. Null if never reached.
export function baselineTimeAtLoss(samples, targetLoss) {
  if (!samples || !samples.length || targetLoss == null) return null;
  let prev = null;
  for (const s of samples) {
    if (!(s.train_loss > 0) || s.train_time_ms == null) continue;
    if (s.train_loss <= targetLoss) {
      if (!prev) return s.train_time_ms; // already at/below at the first valid sample
      const span = prev.train_loss - s.train_loss;
      const f = span > 0 ? (prev.train_loss - targetLoss) / span : 0;
      return prev.train_time_ms + f * (s.train_time_ms - prev.train_time_ms);
    }
    prev = s;
  }
  return null;
}

// Recent loss drop per millisecond (positive when descending). Sparsity-tolerant:
// pairs the latest real-loss sample with one ~windowSteps back, like deltaVsStepsAgo.
export function lossDescentRatePerMs(history, windowSteps = QUALITY_WINDOW_STEPS) {
  if (!history || !history.length) return null;
  let cur = null, past = null;
  for (let i = history.length - 1; i >= 0; i--) {
    const m = history[i];
    if (!(m.train_loss > 0) || m.train_time_ms == null) continue;
    if (!cur) { cur = m; continue; }
    if (m.step <= cur.step - windowSteps) { past = m; break; }
  }
  if (!cur || !past) return null;
  const dt = cur.train_time_ms - past.train_time_ms;
  if (!(dt > 0)) return null;
  return (past.train_loss - cur.train_loss) / dt;
}

// Projected wall-clock at which this run reaches `targetLoss`, holding the
// recent descent rate. currentTime if already there; null if not converging.
export function timeToReachLoss(currentLoss, currentTime, targetLoss, ratePerMs) {
  if (currentLoss == null || currentTime == null || targetLoss == null) return null;
  if (currentLoss <= targetLoss) return currentTime;
  if (ratePerMs == null || ratePerMs <= 0) return null;
  return currentTime + (currentLoss - targetLoss) / ratePerMs;
}

export function smoothCurrentLoss(history, metric) {
  const losses = (history || []).filter((m) => m.train_loss > 0).map((m) => m.train_loss);
  const e = ema(losses, QUALITY_LOSS_EMA_N);
  if (e != null) return e;
  return metric && metric.train_loss > 0 ? metric.train_loss : null;
}

// Single source for the hero card, race banner, and sidebar pace lines.
// `baselineSamples` is the baseline's real-loss metrics sorted by step;
// `baselineSample` is the baseline metric nearest the current step (throughput Δ).
export function qualityRace({
  metric, history, baselineSamples, baselineTotalTimeMs,
  batchTokens, emaMsPerStep, baselineSample,
}) {
  if (!metric) return { state: 'no-data' };
  const msPerStep = emaMsPerStep ?? metric.step_avg_ms ?? null;
  const tps = throughputTps(msPerStep, batchTokens);
  const empty = {
    state: 'no-baseline',
    currentLoss: null, targetLoss: null, leadMs: null,
    projectedTargetTimeMs: null, projectedMarginMs: null,
    throughputTps: tps, msPerStep, baselineMsPerStep: null,
  };
  if (!baselineSamples || !baselineSamples.length) return empty;

  const currentLoss = smoothCurrentLoss(history, metric);
  const currentTime = metric.train_time_ms ?? null;
  if (currentLoss == null || currentTime == null) return empty;

  // Target = baseline's final achieved loss (last sample with a real loss —
  // the literal last row is often a val-only entry with no train_loss).
  let lastValid = null;
  for (let i = baselineSamples.length - 1; i >= 0; i--) {
    if (baselineSamples[i] && baselineSamples[i].train_loss > 0) { lastValid = baselineSamples[i]; break; }
  }
  const targetLoss = lastValid ? lastValid.train_loss : null;
  if (targetLoss == null) return empty;

  // Lead at equal quality: baseline's time to reach my loss, minus mine.
  let leadMs;
  if (currentLoss <= targetLoss) {
    // Already at/below baseline's best — lead measured against its finish.
    leadMs = (baselineTotalTimeMs ?? lastValid.train_time_ms ?? currentTime) - currentTime;
  } else {
    const bt = baselineTimeAtLoss(baselineSamples, currentLoss);
    if (bt == null) return empty;
    leadMs = bt - currentTime;
  }

  const rate = lossDescentRatePerMs(history, QUALITY_WINDOW_STEPS);
  const projectedTargetTimeMs = timeToReachLoss(currentLoss, currentTime, targetLoss, rate);
  const projectedMarginMs = projectedTargetTimeMs != null && baselineTotalTimeMs != null
    ? baselineTotalTimeMs - projectedTargetTimeMs : null;

  return {
    state: leadMs >= 0 ? 'ahead' : 'behind',
    currentLoss, targetLoss, leadMs,
    projectedTargetTimeMs, projectedMarginMs,
    throughputTps: tps, msPerStep,
    baselineMsPerStep: baselineSample?.step_avg_ms ?? null,
  };
}

