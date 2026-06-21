import { throughputTps, lossDescentRatePerMs, timeToReachLoss, smoothCurrentLoss, QUALITY_WINDOW_STEPS } from './quality.js';

// ── Finish-horizon race ──
// Compare runs by their estimated train loss at the configured max-wallclock
// budget (lower wins), not by equal-loss timing. For a behind run we also
// estimate the extra training time needed to cross below the baseline's finish
// loss. Loss-based throughout (no TrainBPB): within one corpus, loss and BPB are
// a constant factor apart, so the ahead/behind verdict is identical.

export const TIE_LOSS = 0.001; // |finish-loss margin| below this reads as too-close-to-call

// Interpolate train_loss at a target wall-clock time across a run's samples.
// Strictly an interpolator: null once the target is past the last valid sample
// (callers extrapolate). Before the first valid sample returns it flagged low.
// Returns { value, atTimeMs, confidence } or null.
export function lossAtTime(samples, targetTimeMs) {
  if (!samples || !samples.length || targetTimeMs == null) return null;
  let prev = null, firstValid = null;
  for (const s of samples) {
    if (!s || !(s.train_loss > 0) || s.train_time_ms == null) continue;
    if (!firstValid) {
      firstValid = s;
      if (targetTimeMs <= s.train_time_ms) {
        return { value: s.train_loss, atTimeMs: s.train_time_ms, confidence: 'low' };
      }
    }
    if (prev && targetTimeMs <= s.train_time_ms) {
      const span = s.train_time_ms - prev.train_time_ms;
      const f = span > 0 ? (targetTimeMs - prev.train_time_ms) / span : 0;
      return { value: prev.train_loss + f * (s.train_loss - prev.train_loss), atTimeMs: targetTimeMs, confidence: 'high' };
    }
    prev = s;
  }
  return null; // target is beyond the last valid sample
}

// Estimate train_loss at the finish horizon. Interpolate when the series brackets
// the horizon; otherwise extrapolate from the latest valid sample at the recent
// descent rate. Returns { value, method, confidence, atTimeMs, ratePerMs };
// value is null ('none') when there's no data or no usable trend to extrapolate.
export function estimateFinishLoss(samples, finishBudgetMs, windowSteps = QUALITY_WINDOW_STEPS) {
  const none = { value: null, method: 'none', confidence: null, atTimeMs: null, ratePerMs: null };
  if (!samples || !samples.length || finishBudgetMs == null) return none;

  const interp = lossAtTime(samples, finishBudgetMs);
  if (interp) {
    return { value: interp.value, method: 'interp', confidence: interp.confidence, atTimeMs: interp.atTimeMs, ratePerMs: null };
  }
  let latest = null;
  for (let i = samples.length - 1; i >= 0; i--) {
    const s = samples[i];
    if (s && s.train_loss > 0 && s.train_time_ms != null) { latest = s; break; }
  }
  if (!latest) return none;
  const rate = lossDescentRatePerMs(samples, windowSteps); // positive when improving
  if (rate == null) return none;
  const projected = Math.max(0, latest.train_loss - rate * (finishBudgetMs - latest.train_time_ms));
  return { value: projected, method: 'extrap', confidence: 'low', atTimeMs: finishBudgetMs, ratePerMs: rate };
}

// Single source for the hero card, race banner, and baseline row. Reuses the
// loss helpers above. `baselineSamples` is the baseline's metrics sorted by step;
// `baselineSample` is the baseline metric nearest the current step (throughput Δ).
export function finishRace({
  metric, history, baselineSamples, currentFinishBudgetMs, baselineFinishBudgetMs,
  batchTokens, emaMsPerStep, baselineSample,
}) {
  if (!metric) return { state: 'no-data' };
  const msPerStep = emaMsPerStep ?? metric.step_avg_ms ?? null;
  const tps = throughputTps(msPerStep, batchTokens);
  const base = {
    state: 'no-baseline',
    currentLoss: null, currentFinishLoss: null, baselineFinishLoss: null, finishMarginLoss: null,
    finishBudgetMs: currentFinishBudgetMs ?? null, baselineFinishBudgetMs: baselineFinishBudgetMs ?? null,
    estimatedWinTimeMs: null, extraTimeToWinMs: null, projectedFinishTimeMs: currentFinishBudgetMs ?? null,
    throughputTps: tps, msPerStep, baselineMsPerStep: baselineSample?.step_avg_ms ?? null,
  };
  if (!baselineSamples || !baselineSamples.length) return base;

  const currentLoss = smoothCurrentLoss(history, metric);
  const currentTime = metric.train_time_ms ?? null;
  base.currentLoss = currentLoss;
  if (currentLoss == null || currentTime == null) return { ...base, state: 'unknown' };

  const activeSamples = (history || []).filter((m) => m && m.train_loss > 0 && m.train_time_ms != null);
  const currentFinishLoss = estimateFinishLoss(activeSamples, currentFinishBudgetMs).value;
  const baselineFinishLoss = estimateFinishLoss(baselineSamples, baselineFinishBudgetMs).value;
  if (currentFinishLoss == null || baselineFinishLoss == null) {
    return { ...base, state: 'unknown', currentFinishLoss, baselineFinishLoss };
  }

  const finishMarginLoss = baselineFinishLoss - currentFinishLoss; // > 0 ⇒ I finish lower ⇒ ahead
  const rate = lossDescentRatePerMs(history, QUALITY_WINDOW_STEPS);
  const estimatedWinTimeMs = timeToReachLoss(currentLoss, currentTime, baselineFinishLoss, rate);
  const extraTimeToWinMs = (estimatedWinTimeMs != null && currentFinishBudgetMs != null && estimatedWinTimeMs > currentFinishBudgetMs)
    ? estimatedWinTimeMs - currentFinishBudgetMs : null;

  const state = Math.abs(finishMarginLoss) < TIE_LOSS ? 'tied' : (finishMarginLoss > 0 ? 'ahead' : 'behind');
  return {
    ...base, state, currentFinishLoss, baselineFinishLoss, finishMarginLoss,
    estimatedWinTimeMs, extraTimeToWinMs,
  };
}

