// Unit tests for the derivation module.
// Run: node --test podterm/static/js/derive.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ema, etaMs, costSoFar, projectedTotalCost, elapsedWallMs,
  baselineAtStep, paceDelta, requiredPaceMs, projectedFinishMarginMs,
  raceStatus, deltaVsStepsAgo, evalDelta, parseGpuMemGiB,
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

test('spec: 192.6 ms vs required 199.9 ms over 16,883 steps -> margin ~123 s', () => {
  const m = projectedFinishMarginMs(199.9, 192.6, 16883);
  assert.ok(Math.abs(m / 1000 - 123.2) < 0.5, `got ${m / 1000} s`);
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

// ── paceDelta ──
test('paceDelta: negative deltas mean ahead', () => {
  const p = paceDelta(
    { step_avg_ms: 192.6, train_time_ms: 600_000 },
    { step_avg_ms: 200.0, train_time_ms: 600_200 },
  );
  assert.ok(Math.abs(p.perStepMs - -7.4) < 1e-9);
  assert.equal(p.cumulativeMs, -200);
});

test('paceDelta: missing fields -> null', () => {
  assert.equal(paceDelta({ step_avg_ms: 192 }, { step_avg_ms: 200 }), null);
  assert.equal(paceDelta(null, {}), null);
});

// ── requiredPaceMs ──
test('requiredPaceMs: basic and zero-remaining guard', () => {
  // Baseline finished in 60s; we are 30s in with 150 steps left -> 200 ms/step.
  assert.equal(requiredPaceMs(60_000, 30_000, 150), 200);
  assert.equal(requiredPaceMs(60_000, 30_000, 0), null);
});

// ── raceStatus ──
const baselineSamples = {
  250: { step: 250, step_avg_ms: 200, train_time_ms: 50_000 },
  500: { step: 500, step_avg_ms: 200, train_time_ms: 100_000 },
};

test('raceStatus: ahead when cumulative time is lower', () => {
  const r = raceStatus({
    metric: { step: 500, total_steps: 1000, step_avg_ms: 192.6, train_time_ms: 96_300 },
    baselineByStep: baselineSamples,
    baselineSteps: [250, 500],
    baselineTotalTimeMs: 200_000,
    emaMs: 192.6,
  });
  assert.equal(r.state, 'ahead');
  assert.equal(r.cumulativeMs, -3700);
  assert.ok(Math.abs(r.perStepMs - -7.4) < 1e-9);
  // required = (200000 - 96300) / 500 = 207.4; margin = (207.4 - 192.6) * 500
  assert.ok(Math.abs(r.requiredMs - 207.4) < 1e-9);
  assert.ok(Math.abs(r.projectedMarginMs - 7400) < 1e-6);
});

test('raceStatus: no baseline -> no-baseline state, no numbers', () => {
  const r = raceStatus({ metric: { step: 1 }, baselineSteps: [], baselineByStep: {} });
  assert.equal(r.state, 'no-baseline');
  assert.equal(r.cumulativeMs, null);
});

test('raceStatus: no metric -> no-data', () => {
  assert.equal(raceStatus({ metric: null }).state, 'no-data');
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
