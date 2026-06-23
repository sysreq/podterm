import { throughputTps, lossDescentRatePerMs, timeToReachLoss, smoothCurrentLoss, QUALITY_WINDOW_STEPS } from './quality.js';

// ── Finish-horizon race ──
// Compare runs by their estimated train loss at the configured max-wallclock
// budget (lower wins), not by equal-loss timing. For a behind run we also
// estimate the extra training time needed to cross below the baseline's finish
// loss. Loss-based throughout (no TrainBPB): within one corpus, loss and BPB are
// a constant factor apart, so the ahead/behind verdict is identical.

export const TIE_LOSS = 0.001; // |finish-loss margin| below this reads as too-close-to-call

// The loss curve decelerates (convex), so the finish projection fits loss vs
// ln(time) — roughly linear across a training run — and extrapolates that to the
// budget. A linear-in-*time* fit of the steep early descent shoots well below zero
// (clamping to ~0) and makes a clearly-behind run look like it finishes near 0.
//
// Fit only the most recent fraction of elapsed time so the steep launch transient
// is excluded and the slope reflects the current (flattening) convergence regime,
// not the early plunge.
const RECENT_TREND_FRACTION = 0.5;
// Don't declare an ahead/behind finish verdict until this fraction of the budget has
// elapsed: before then we'd extrapolate across most of the run from a still-steep
// curve, which isn't trustworthy. The sidebar still shows current standing meanwhile.
export const MIN_PROJECTION_FRACTION = 0.2;

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

// Least-squares line of train_loss vs ln(train_time_ms) over the most recent
// `frac` of elapsed time. ln-time linearises the decelerating loss curve, and the
// recent-fraction window drops the steep launch transient, so the slope reflects
// the current convergence regime. Returns { slope, intercept, n } (slope in
// loss-per-ln-ms, negative while improving) or null with < 2 usable points.
// Fitting the window — not the latest step — also smooths per-step variance, so a
// single lucky/unlucky step can't swing the finish projection.
function recentLogTimeTrend(samples, frac = RECENT_TREND_FRACTION) {
  let latestTime = null;
  for (let i = samples.length - 1; i >= 0; i--) {
    const s = samples[i];
    if (s && s.train_loss > 0 && s.train_time_ms > 0) { latestTime = s.train_time_ms; break; }
  }
  if (latestTime == null) return null;
  const threshold = frac * latestTime;
  let n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (const s of samples) {
    if (!s || !(s.train_loss > 0) || !(s.train_time_ms > 0) || s.train_time_ms < threshold) continue;
    const x = Math.log(s.train_time_ms), y = s.train_loss;
    n += 1; sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  if (n < 2) return null;
  const denom = n * sxx - sx * sx;
  if (!(Math.abs(denom) > 0)) return null;
  const slope = (n * sxy - sx * sy) / denom;
  return { slope, intercept: (sy - slope * sx) / n, n };
}

// Estimate train_loss at the finish horizon. Interpolate when the series brackets
// the horizon; otherwise extrapolate the recent ln(time) trend to the horizon —
// decelerating like the real curve, fit over the recent window (never anchored on
// the single latest noisy step). Returns { value, method, confidence, atTimeMs,
// ratePerMs }; value is null ('none') when there's no data or no usable trend.
export function estimateFinishLoss(samples, finishBudgetMs, windowSteps = QUALITY_WINDOW_STEPS) {
  const none = { value: null, method: 'none', confidence: null, atTimeMs: null, ratePerMs: null };
  if (!samples || !samples.length || finishBudgetMs == null || !(finishBudgetMs > 0)) return none;

  const interp = lossAtTime(samples, finishBudgetMs);
  if (interp) {
    return { value: interp.value, method: 'interp', confidence: interp.confidence, atTimeMs: interp.atTimeMs, ratePerMs: null };
  }
  const trend = recentLogTimeTrend(samples);
  if (trend) {
    const projected = Math.max(0, trend.intercept + trend.slope * Math.log(finishBudgetMs));
    return { value: projected, method: 'trend', confidence: 'low', atTimeMs: finishBudgetMs, ratePerMs: null };
  }
  // Too few points to fit a trend: hold the latest sample at the recent descent rate.
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
  // Too early / degenerate to call: before MIN_PROJECTION_FRACTION of the budget the
  // projection extrapolates across most of the run, and a clamp to ≤0 means the steep
  // curve overshot. Either way report 'unknown' rather than a confident, likely-wrong
  // verdict — the curve hasn't settled enough to know who finishes lower.
  const elapsedFrac = currentFinishBudgetMs > 0 ? currentTime / currentFinishBudgetMs : 1;
  if (currentFinishLoss <= 0 || elapsedFrac < MIN_PROJECTION_FRACTION) {
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

