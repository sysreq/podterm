// Unit tests for the derivation module.
// Run: node --test podterm/static/js/derive.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ema, etaMs, costSoFar, projectedTotalCost, elapsedWallMs,
  baselineAtStep, throughputTps, baselineTimeAtLoss, lossDescentRatePerMs,
  timeToReachLoss, qualityRace, deltaVsStepsAgo, evalDelta, parseGpuMemGiB,
} from './derive.js';

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

// ── deltaVsStepsAgo ──
const sparseHistory = [
  { step: 250, train_loss: 4.0 },
  { step: 500, train_loss: 3.5 },
  { step: 750, train_loss: 0.0 },  // val-only row — must be skipped for train_loss
  { step: 750, val_bpb: 1.5 },
  { step: 1000, train_loss: 3.2 },
];

test('deltaVsStepsAgo: sparse 250-step cadence resolves to real samples', () => {
  const d = deltaVsStepsAgo(sparseHistory, 'train_loss', 1000, 100);
  assert.equal(d.current, 3.2);
  // nearest sample <= 900 with a real train_loss is step 500 (750 is val-only)
  assert.equal(d.delta, 3.2 - 3.5);
  assert.equal(d.stepsSpanned, 500);
});

test('deltaVsStepsAgo: not enough history -> null', () => {
  assert.equal(deltaVsStepsAgo([{ step: 250, train_loss: 4 }], 'train_loss', 250, 100), null);
});

// ── evalDelta ──
test('evalDelta: delta vs previous eval window', () => {
  const d = evalDelta([
    { step: 500, val_bpb: 1.60 },
    { step: 1000, val_bpb: 1.52 },
  ]);
  assert.equal(d.current, 1.52);
  assert.ok(Math.abs(d.delta - -0.08) < 1e-9);
});

test('evalDelta: single eval -> no delta yet', () => {
  const d = evalDelta([{ step: 500, val_bpb: 1.6 }]);
  assert.equal(d.current, 1.6);
  assert.equal(d.delta, null);
});

// ── metadata ──
test('parseGpuMemGiB: extracts from GPU name, null when absent', () => {
  assert.equal(parseGpuMemGiB('NVIDIA H100 80GB HBM3'), 80);
  assert.equal(parseGpuMemGiB('NVIDIA GeForce RTX 4090'), null);
  assert.equal(parseGpuMemGiB(null), null);
});

// ── cost ──
test('projectedTotalCost: elapsed + ETA at the hourly rate', () => {
  // 30 min elapsed + 30 min remaining at $2/hr -> $2
  assert.ok(Math.abs(projectedTotalCost(1_800_000, 1_800_000, 2) - 2) < 1e-9);
});

test('elapsedWallMs: parses ISO and clamps negatives to zero', () => {
  const now = Date.parse('2026-06-12T10:10:00');
  assert.equal(elapsedWallMs('2026-06-12T10:00:00', now), 600_000);
  assert.equal(elapsedWallMs('2026-06-12T11:00:00', now), 0);
  assert.equal(elapsedWallMs('garbage', now), null);
});
