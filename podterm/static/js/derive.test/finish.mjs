import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ema, etaMs, costSoFar, projectedTotalCost, elapsedWallMs,
  baselineAtStep, throughputTps, baselineTimeAtLoss, lossDescentRatePerMs,
  timeToReachLoss, qualityRace, deltaVsStepsAgo, evalDelta, parseGpuMemGiB,
  lossAtTime, estimateFinishLoss, finishRace, configFinishBudgetMs, TIE_LOSS,
} from '../derive.js';

// ── Finish-horizon race ──
// A run whose loss falls 4.0 -> 3.0 -> 2.5 at 50s, 100s, 150s.
const finSamples = [
  { step: 250, train_loss: 4.0, train_time_ms: 50_000, step_avg_ms: 180 },
  { step: 500, train_loss: 3.0, train_time_ms: 100_000, step_avg_ms: 180 },
  { step: 750, train_loss: 2.5, train_time_ms: 150_000, step_avg_ms: 180 },
];

test('lossAtTime: interpolates loss at a bracketed time', () => {
  // 75s sits halfway between 50s (4.0) and 100s (3.0) -> 3.5
  const r = lossAtTime(finSamples, 75_000);
  assert.equal(r.value, 3.5);
  assert.equal(r.confidence, 'high');
  // exact last-sample hit
  assert.equal(lossAtTime(finSamples, 150_000).value, 2.5);
});

test('lossAtTime: before first -> first loss (low conf); after last -> null', () => {
  const before = lossAtTime(finSamples, 25_000);
  assert.equal(before.value, 4.0);
  assert.equal(before.confidence, 'low');
  assert.equal(lossAtTime(finSamples, 200_000), null);
});

test('estimateFinishLoss: interpolates inside the observed series', () => {
  const r = estimateFinishLoss(finSamples, 75_000);
  assert.equal(r.method, 'interp');
  assert.equal(r.value, 3.5);
});

test('estimateFinishLoss: extrapolates at the recent rate beyond the horizon', () => {
  // rate over 500 steps = (4.0-2.5)/100000ms = 1.5e-5; from 2.5 at 150s out to 200s
  // -> 2.5 - 1.5e-5 * 50000 = 1.75
  const r = estimateFinishLoss(finSamples, 200_000);
  assert.equal(r.method, 'extrap');
  assert.ok(Math.abs(r.value - 1.75) < 1e-9, `got ${r.value}`);
});

test('estimateFinishLoss: no usable trend -> none (null value)', () => {
  const r = estimateFinishLoss([finSamples[2]], 200_000);
  assert.equal(r.method, 'none');
  assert.equal(r.value, null);
});

test('finishRace: ahead when projected to finish lower than baseline', () => {
  const baseSlow = [
    { step: 250, train_loss: 4.0, train_time_ms: 50_000 },
    { step: 500, train_loss: 3.6, train_time_ms: 100_000 },
    { step: 750, train_loss: 3.4, train_time_ms: 150_000 },
  ];
  const r = finishRace({
    metric: finSamples[2], history: finSamples, baselineSamples: baseSlow,
    currentFinishBudgetMs: 200_000, baselineFinishBudgetMs: 150_000,
    batchTokens: 524288, emaMsPerStep: 180, baselineSample: { step_avg_ms: 200 },
  });
  assert.equal(r.state, 'ahead');
  assert.ok(Math.abs(r.currentFinishLoss - 1.75) < 1e-9);
  assert.equal(r.baselineFinishLoss, 3.4);
  assert.ok(r.finishMarginLoss > 0);
  assert.ok(r.throughputTps > 0);
  assert.equal(r.baselineMsPerStep, 200);
});

test('finishRace: behind + improving -> time-to-win beyond budget', () => {
  const slowImprove = [
    { step: 250, train_loss: 4.0, train_time_ms: 50_000, step_avg_ms: 180 },
    { step: 500, train_loss: 3.8, train_time_ms: 100_000, step_avg_ms: 180 },
    { step: 750, train_loss: 3.7, train_time_ms: 150_000, step_avg_ms: 180 },
  ];
  const baseFast = [
    { step: 250, train_loss: 4.0, train_time_ms: 50_000 },
    { step: 500, train_loss: 3.5, train_time_ms: 100_000 },
    { step: 750, train_loss: 3.4, train_time_ms: 150_000 },
  ];
  const r = finishRace({
    metric: slowImprove[2], history: slowImprove, baselineSamples: baseFast,
    currentFinishBudgetMs: 160_000, baselineFinishBudgetMs: 150_000,
    batchTokens: 524288, emaMsPerStep: 180,
  });
  assert.equal(r.state, 'behind');
  assert.ok(r.estimatedWinTimeMs > r.finishBudgetMs, `win ${r.estimatedWinTimeMs}`);
  assert.ok(r.extraTimeToWinMs > 0);
});

test('finishRace: tied within the loss band', () => {
  const aTie = [
    { step: 250, train_loss: 3.2, train_time_ms: 50_000, step_avg_ms: 180 },
    { step: 500, train_loss: 3.0005, train_time_ms: 100_000, step_avg_ms: 180 },
  ];
  const bTie = [
    { step: 250, train_loss: 3.2, train_time_ms: 50_000 },
    { step: 500, train_loss: 3.0, train_time_ms: 100_000 },
  ];
  const r = finishRace({
    metric: aTie[1], history: aTie, baselineSamples: bTie,
    currentFinishBudgetMs: 100_000, baselineFinishBudgetMs: 100_000,
    batchTokens: 524288, emaMsPerStep: 180,
  });
  assert.equal(r.state, 'tied');
  assert.ok(Math.abs(r.finishMarginLoss) < TIE_LOSS);
});

test('finishRace: no baseline / no metric / unprojectable -> graceful states', () => {
  assert.equal(finishRace({ metric: null }).state, 'no-data');
  assert.equal(finishRace({
    metric: finSamples[2], history: finSamples, baselineSamples: [],
    batchTokens: 524288, emaMsPerStep: 180,
  }).state, 'no-baseline');
  // baseline present but the active run can't be projected yet (single sample, horizon ahead)
  const single = finSamples[2];
  const r = finishRace({
    metric: single, history: [single],
    baselineSamples: [{ step: 250, train_loss: 3.0, train_time_ms: 100_000 }],
    currentFinishBudgetMs: 999_999, baselineFinishBudgetMs: 100_000,
    batchTokens: 524288, emaMsPerStep: 180,
  });
  assert.equal(r.state, 'unknown');
  assert.equal(r.currentFinishLoss, null);
});

test('configFinishBudgetMs: seconds -> ms; null when absent/unparseable', () => {
  assert.equal(configFinishBudgetMs({ config_json: JSON.stringify({ time_budget: 600 }) }), 600_000);
  assert.equal(configFinishBudgetMs({ config_json: JSON.stringify({ branch: 'x' }) }), null); // pre-budget run
  assert.equal(configFinishBudgetMs({ config_json: '{not json' }), null);
  assert.equal(configFinishBudgetMs({}), null);
  assert.equal(configFinishBudgetMs(null), null);
});

