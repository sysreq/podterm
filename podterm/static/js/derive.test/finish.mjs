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

test('estimateFinishLoss: extrapolates the ln(time) trend beyond the horizon', () => {
  // Recent-half window = the two latest points (100s→3.0, 150s→2.5). Fit loss vs
  // ln(t): slope = -0.5/ln(1.5); at the 200s horizon →
  // 2.5 + slope·ln(200/150) ≈ 2.145. Decelerating (ln-time), not linear-in-time.
  const expected = 2.5 + (-0.5 / Math.log(1.5)) * Math.log(200 / 150);
  const r = estimateFinishLoss(finSamples, 200_000);
  assert.equal(r.method, 'trend');
  assert.ok(Math.abs(r.value - expected) < 1e-9, `got ${r.value}, want ${expected}`);
});

test('estimateFinishLoss: a single noisy last step barely moves the projection', () => {
  // A dense, smooth descent (20 samples). Perturbing only the final sample by 0.5
  // moves the trend-fit projection by a fraction of that — the old latest-point
  // extrapolation would move ~1:1 with the noise. This is the whole point: a good
  // (or bad) step shouldn't flip the verdict.
  const dense = [];
  for (let k = 1; k <= 20; k++) dense.push({ step: k * 100, train_loss: 3.0 - 0.04 * k, train_time_ms: k * 20_000 });
  const clean = estimateFinishLoss(dense, 410_000).value;
  const noisy = [...dense.slice(0, 19), { ...dense[19], train_loss: dense[19].train_loss - 0.5 }];
  const perturbed = estimateFinishLoss(noisy, 410_000).value;
  assert.ok(Math.abs(perturbed - clean) < 0.2, `swing too large: ${clean} -> ${perturbed}`);
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
  assert.ok(Math.abs(r.currentFinishLoss - (2.5 + (-0.5 / Math.log(1.5)) * Math.log(200 / 150))) < 1e-9);
  assert.equal(r.baselineFinishLoss, 3.4);
  assert.ok(r.finishMarginLoss > 0);
  assert.ok(r.throughputTps > 0);
  assert.equal(r.baselineMsPerStep, 200);
});

test('finishRace: a run worse than baseline throughout reads behind, not ahead', () => {
  // Regression: the current run is above the baseline at every step (worse). A
  // linear-in-time extrapolation of its descent overshoots and wrongly flips it to
  // "ahead"; the ln(time) trend keeps the projection realistic → behind.
  const budget = 600_000;
  const current = [
    { step: 500, train_loss: 3.4, train_time_ms: 100_000, step_avg_ms: 200 },
    { step: 1000, train_loss: 3.0, train_time_ms: 200_000, step_avg_ms: 200 },
    { step: 1500, train_loss: 2.85, train_time_ms: 300_000, step_avg_ms: 200 },
    { step: 2000, train_loss: 2.8, train_time_ms: 400_000, step_avg_ms: 200 },
    { step: 2250, train_loss: 2.78, train_time_ms: 450_000, step_avg_ms: 200 },
  ];
  const baseline = [
    { step: 500, train_loss: 3.0, train_time_ms: 100_000 },
    { step: 1000, train_loss: 2.7, train_time_ms: 200_000 },
    { step: 1500, train_loss: 2.6, train_time_ms: 300_000 },
    { step: 2000, train_loss: 2.55, train_time_ms: 400_000 },
    { step: 2500, train_loss: 2.52, train_time_ms: 500_000 },
    { step: 3000, train_loss: 2.5, train_time_ms: 600_000 },
  ];
  const r = finishRace({
    metric: current[current.length - 1], history: current, baselineSamples: baseline,
    currentFinishBudgetMs: budget, baselineFinishBudgetMs: budget,
    batchTokens: 524288, emaMsPerStep: 200,
  });
  assert.equal(r.state, 'behind');
  assert.ok(r.currentFinishLoss > 0, `degenerate projection: ${r.currentFinishLoss}`);
  assert.ok(r.currentFinishLoss > r.baselineFinishLoss, `current ${r.currentFinishLoss} should finish above baseline ${r.baselineFinishLoss}`);
});

test('finishRace: too early in the budget -> unknown (no confident verdict)', () => {
  // Only 5% of the budget elapsed: extrapolating the still-steep launch isn't
  // trustworthy, so no ahead/behind claim.
  const current = [
    { step: 50, train_loss: 7.0, train_time_ms: 10_000, step_avg_ms: 200 },
    { step: 100, train_loss: 5.0, train_time_ms: 20_000, step_avg_ms: 200 },
    { step: 150, train_loss: 3.8, train_time_ms: 30_000, step_avg_ms: 200 },
  ];
  const baseline = [
    { step: 1500, train_loss: 2.6, train_time_ms: 300_000 },
    { step: 3000, train_loss: 2.5, train_time_ms: 600_000 },
  ];
  const r = finishRace({
    metric: current[current.length - 1], history: current, baselineSamples: baseline,
    currentFinishBudgetMs: 600_000, baselineFinishBudgetMs: 600_000,
    batchTokens: 524288, emaMsPerStep: 200,
  });
  assert.equal(r.state, 'unknown');
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

