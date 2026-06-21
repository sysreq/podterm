// ── Baseline race math ──
// Last baseline sample at or before `step` (metrics are sparse — every ~250 steps).
export function baselineAtStep(baselineByStep, sortedSteps, step) {
  if (!sortedSteps || !sortedSteps.length || step == null) return null;
  let lo = 0, hi = sortedSteps.length - 1, found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (sortedSteps[mid] <= step) { found = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return found >= 0 ? baselineByStep[sortedSteps[found]] : null;
}

