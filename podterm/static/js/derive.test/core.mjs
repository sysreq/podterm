import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ema, etaMs, costSoFar, projectedTotalCost, elapsedWallMs,
  baselineAtStep, throughputTps, baselineTimeAtLoss, lossDescentRatePerMs,
  timeToReachLoss, qualityRace, deltaVsStepsAgo, evalDelta, parseGpuMemGiB,
  lossAtTime, estimateFinishLoss, finishRace, configFinishBudgetMs, TIE_LOSS,
} from '../derive.js';

// ── The three spec sanity cases ──
test('spec: 16,883 steps remaining at 192.6 ms/step -> ETA ~54 min', () => {
  const eta = etaMs(16883, 192.6);
  assert.ok(Math.abs(eta / 60000 - 54.2) < 0.1, `got ${eta / 60000} min`);
});

test('spec: $3.29/hr with ~10 min elapsed -> cost ~$0.55', () => {
  const c = costSoFar(10 * 60 * 1000, 3.29);
  assert.ok(Math.abs(c - 0.548) < 0.005, `got $${c}`);
});



// ── ema ──
test('ema: empty/null input -> null', () => {
  assert.equal(ema([], 50), null);
  assert.equal(ema(null, 50), null);
});

test('ema: constant series -> the constant', () => {
  assert.ok(Math.abs(ema([5, 5, 5, 5], 50) - 5) < 1e-9);
});

test('ema: weights recent values more than old ones', () => {
  const rising = ema([100, 100, 100, 200], 50);
  assert.ok(rising > 100 && rising < 200);
  const avg = (100 + 100 + 100 + 200) / 4;
  assert.ok(rising > avg, 'EMA should lean toward the recent 200');
});

// ── baselineAtStep ──
const byStep = { 250: { step: 250, v: 'a' }, 500: { step: 500, v: 'b' }, 750: { step: 750, v: 'c' } };
const steps = [250, 500, 750];

test('baselineAtStep: before first sample -> null', () => {
  assert.equal(baselineAtStep(byStep, steps, 100), null);
});

test('baselineAtStep: exact hit', () => {
  assert.equal(baselineAtStep(byStep, steps, 500).v, 'b');
});

test('baselineAtStep: between samples -> earlier sample', () => {
  assert.equal(baselineAtStep(byStep, steps, 600).v, 'b');
});

test('baselineAtStep: past the last sample -> last sample', () => {
  assert.equal(baselineAtStep(byStep, steps, 99999).v, 'c');
});

// ── throughputTps ──
test('throughputTps: tokens per second from step time + token budget', () => {
  // 524288 tokens/step at 200 ms/step -> ~2.62M tok/s
  assert.ok(Math.abs(throughputTps(200, 524288) - 2_621_440) < 1);
  assert.equal(throughputTps(200, null), null);
  assert.equal(throughputTps(0, 524288), null);
});

// ── baselineTimeAtLoss ──
// Baseline: loss falls 4.0 -> 3.0 -> 2.5 at 50s, 100s, 150s.
const blSamples = [
  { step: 250, train_loss: 4.0, train_time_ms: 50_000 },
  { step: 500, train_loss: 3.0, train_time_ms: 100_000 },
  { step: 750, train_loss: 2.5, train_time_ms: 150_000 },
];

test('baselineTimeAtLoss: interpolates time at a loss between samples', () => {
  // loss 3.5 sits halfway between 4.0 and 3.0 -> halfway in time (75s)
  assert.equal(baselineTimeAtLoss(blSamples, 3.5), 75_000);
  // exact sample hit
  assert.equal(baselineTimeAtLoss(blSamples, 3.0), 100_000);
});

test('baselineTimeAtLoss: never reached -> null; already below at first -> first time', () => {
  assert.equal(baselineTimeAtLoss(blSamples, 2.0), null);
  assert.equal(baselineTimeAtLoss(blSamples, 5.0), 50_000);
});

// ── lossDescentRatePerMs / timeToReachLoss ──
test('lossDescentRatePerMs: recent loss drop per ms (positive when descending)', () => {
  const hist = [
    { step: 0, train_loss: 4.0, train_time_ms: 0 },
    { step: 500, train_loss: 3.0, train_time_ms: 100_000 },
  ];
  // (4.0 - 3.0) / 100000 ms = 1e-5 loss/ms
  assert.ok(Math.abs(lossDescentRatePerMs(hist, 500) - 1e-5) < 1e-12);
  assert.equal(lossDescentRatePerMs([{ step: 0, train_loss: 4, train_time_ms: 0 }], 500), null);
});

test('timeToReachLoss: projects forward, guards non-convergence', () => {
  // at loss 3.0, time 100s, target 2.5, rate 1e-5 -> +50s -> 150s
  assert.equal(timeToReachLoss(3.0, 100_000, 2.5, 1e-5), 150_000);
  assert.equal(timeToReachLoss(2.4, 100_000, 2.5, 1e-5), 100_000); // already past target
  assert.equal(timeToReachLoss(3.0, 100_000, 2.5, 0), null);       // flat -> null
});

