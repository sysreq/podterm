// ── Time / cost ──
export function etaMs(remainingSteps, msPerStep) {
  if (remainingSteps == null || msPerStep == null || remainingSteps < 0) return null;
  return remainingSteps * msPerStep;
}

export function costSoFar(elapsedWallMs, costPerHr) {
  if (elapsedWallMs == null || costPerHr == null) return null;
  return (elapsedWallMs / 3_600_000) * costPerHr;
}

export function projectedTotalCost(elapsedWallMs, eta, costPerHr) {
  if (elapsedWallMs == null || eta == null || costPerHr == null) return null;
  return ((elapsedWallMs + eta) / 3_600_000) * costPerHr;
}

export function elapsedWallMs(startedAtIso, nowMs) {
  if (!startedAtIso || nowMs == null) return null;
  const t = Date.parse(startedAtIso);
  if (Number.isNaN(t)) return null;
  return Math.max(0, nowMs - t);
}

