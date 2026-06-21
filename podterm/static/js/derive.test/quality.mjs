import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ema, etaMs, costSoFar, projectedTotalCost, elapsedWallMs,
  baselineAtStep, throughputTps, baselineTimeAtLoss, lossDescentRatePerMs,
  timeToReachLoss, qualityRace, deltaVsStepsAgo, evalDelta, parseGpuMemGiB,
  lossAtTime, estimateFinishLoss, finishRace, configFinishBudgetMs, TIE_LOSS,
} from '../derive.js';

const blSamples = [
  { step: 250, train_loss: 4.0, train_time_ms: 50_000 },
  { step: 500, train_loss: 3.0, train_time_ms: 100_000 },
  { step: 750, train_loss: 2.5, train_time_ms: 150_000 },
];

// ── qualityRace ──
const myHistory = [
  { step: 0, train_loss: 4.0, train_time_ms: 0, step_avg_ms: 180 },
  { step: 500, train_loss: 3.5, train_time_ms: 45_000, step_avg_ms: 180 },
];

test('qualityRace: ahead when reaching the loss faster than baseline', () => {
  // I am at loss 3.5 at 45s; baseline hit 3.5 at 75s -> ahead by 30s.
  const r = qualityRace({
    metric: myHistory[1],
    history: myHistory,
    baselineSamples: blSamples,
    baselineTotalTimeMs: 150_000,
    batchTokens: 524288,
    emaMsPerStep: 180,
    baselineSample: { step_avg_ms: 200 },
  });
  assert.equal(r.state, 'ahead');
  assert.ok(r.leadMs > 0, `leadMs ${r.leadMs}`);
  assert.equal(r.targetLoss, 2.5);
  assert.ok(r.throughputTps > 0);
  assert.ok(Math.abs(r.msPerStep - 180) < 1e-9);
  assert.equal(r.baselineMsPerStep, 200);
});

test('qualityRace: no baseline -> no-baseline but still reports throughput', () => {
  const r = qualityRace({
    metric: myHistory[1], history: myHistory, baselineSamples: [],
    batchTokens: 524288, emaMsPerStep: 180,
  });
  assert.equal(r.state, 'no-baseline');
  assert.equal(r.leadMs, null);
  assert.ok(r.throughputTps > 0);
});

test('qualityRace: null batchTokens -> ms/step only, no tps', () => {
  const r = qualityRace({
    metric: myHistory[1], history: myHistory, baselineSamples: blSamples,
    baselineTotalTimeMs: 150_000, batchTokens: null, emaMsPerStep: 180,
  });
  assert.equal(r.throughputTps, null);
  assert.ok(Math.abs(r.msPerStep - 180) < 1e-9);
});

test('qualityRace: no metric -> no-data', () => {
  assert.equal(qualityRace({ metric: null }).state, 'no-data');
});

