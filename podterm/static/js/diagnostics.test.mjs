// Unit tests for the pure Model Health selectors. Run: node --test podterm/static/js/diagnostics.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  latestSnapshot, snapshotByStep, previousSnapshot,
  headlineByGroup, topHealthIssues, groupSummary, referenceDoc,
} from './diagnostics.js';

const metric = (id, group, band, value, tier = 1) =>
  ({ id, label: id, value, unit: 'x', band, group, tier, thresholds: { kind: 'hi', warn: 1, bad: 2 }, why: 'because' });

// step 100: all good. step 200: a warn + a bad. plus a trailing null-diag entry (mid-run, no result yet).
const snap = (step, headline, sections = []) => ({
  step, created_at: `2026-06-20T00:0${step / 100}:00+00:00`, status: 'ok',
  diag: { health: { overall: 'ok', counts: {}, headline }, sections },
});

const h100 = [metric('dead_neurons', 'capacity', 'good', 0.1), metric('grad_ratio', 'grad', 'good', 0.2, 2),
  metric('eff_rank', 'arch', 'good', 0.3, 1)];
const h200 = [metric('dead_neurons', 'capacity', 'bad', 0.9), metric('grad_ratio', 'grad', 'warn', 0.5, 2),
  metric('eff_rank', 'arch', 'good', 0.4, 1)];

const sec = (name, status, rows = {}) => ({ name, status, rows });
const HISTORY = [
  snap(100, h100, [sec('gradients', 'ok', { summary: { ratio: 100 } })]),
  snap(200, h200, [sec('gradients', 'warn', { summary: { ratio: 5000 } })]),
  { step: 250, created_at: 'x', status: 'ok', diag: null }, // not yet computed
];

test('latestSnapshot skips entries without .diag', () => {
  assert.equal(latestSnapshot(HISTORY).step, 200);
  assert.equal(latestSnapshot([]), null);
});

test('snapshotByStep finds by step or returns null', () => {
  assert.equal(snapshotByStep(HISTORY, 100).step, 100);
  assert.equal(snapshotByStep(HISTORY, 999), null);
});

test('previousSnapshot returns the prior valid snapshot', () => {
  assert.equal(previousSnapshot(HISTORY, 200).step, 100);
  assert.equal(previousSnapshot(HISTORY, 100), null); // none before the first
});

test('headlineByGroup buckets by group key (grad/arch, not gradients/architecture)', () => {
  const g = headlineByGroup(snap(200, h200));
  assert.deepEqual(Object.keys(g), ['capacity', 'grad', 'arch']);
  assert.equal(g.grad[0].id, 'grad_ratio');
});

test('topHealthIssues: warn/bad first (bad before warn)', () => {
  const issues = topHealthIssues(snap(200, h200));
  assert.deepEqual(issues.map((m) => m.id), ['dead_neurons', 'grad_ratio']);
});

test('topHealthIssues: healthy fallback = top-3 good by tier', () => {
  const issues = topHealthIssues(snap(100, h100));
  assert.equal(issues.length, 3);
  assert.ok(issues.every((m) => m.band === 'good'));
  // tier 1 metrics sort ahead of the tier-2 grad_ratio
  assert.equal(issues[issues.length - 1].id, 'grad_ratio');
});

test('groupSummary: worst band + per-band counts per group', () => {
  const gs = groupSummary(snap(200, h200));
  const cap = gs.find((g) => g.group === 'capacity');
  assert.equal(cap.worst, 'bad');
  assert.equal(cap.counts.bad, 1);
  assert.equal(cap.label, 'Capacity');
  const grad = gs.find((g) => g.group === 'grad');
  assert.equal(grad.worst, 'warn');
});

test('referenceDoc resolves per diff mode', () => {
  assert.equal(referenceDoc(HISTORY, 200, 'previous', []).sections[0].rows.summary.ratio, 100);
  assert.equal(referenceDoc(HISTORY, 100, 'previous', []), null); // no prior
  assert.equal(referenceDoc(HISTORY, 200, 'absolute', HISTORY), null);
  const baseline = [snap(50, h100, [sec('gradients', 'ok', { summary: { ratio: 42 } })])];
  assert.equal(referenceDoc(HISTORY, 200, 'baseline', baseline).sections[0].rows.summary.ratio, 42);
  assert.equal(referenceDoc(HISTORY, 200, 'baseline', []), null); // no baseline data
});
