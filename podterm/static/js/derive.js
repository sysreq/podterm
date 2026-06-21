// Public facade for dashboard derivation helpers.
export { ETA_EMA_N, AVG_EMA_N, ema } from './derive/ema.js';
export { etaMs, costSoFar, projectedTotalCost, elapsedWallMs } from './derive/time.js';
export { baselineAtStep } from './derive/baseline.js';
export {
  QUALITY_WINDOW_STEPS, throughputTps, baselineTimeAtLoss,
  lossDescentRatePerMs, timeToReachLoss, qualityRace,
} from './derive/quality.js';
export { TIE_LOSS, lossAtTime, estimateFinishLoss, finishRace } from './derive/finish.js';
export { deltaVsStepsAgo, evalDelta, parseGpuMemGiB, configFinishBudgetMs } from './derive/stats.js';
