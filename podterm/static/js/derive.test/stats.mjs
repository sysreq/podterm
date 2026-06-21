import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ema, etaMs, costSoFar, projectedTotalCost, elapsedWallMs,
  baselineAtStep, throughputTps, baselineTimeAtLoss, lossDescentRatePerMs,
  timeToReachLoss, qualityRace, deltaVsStepsAgo, evalDelta, parseGpuMemGiB,
  lossAtTime, estimateFinishLoss, finishRace, configFinishBudgetMs, TIE_LOSS,
} from '../derive.js';

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
