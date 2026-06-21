import { app, getPodState } from '../../state.js';
import * as d from '../../derive.js';
import { fmtClock, fmtDelta, fmtDuration, fmtInt, fmtMs, fmtTps } from '../../format.js';
import { updateBootPanel } from '../boot.js';
import { updateBaselineRow, updateRaceBanner } from '../race.js';
import { buildKpiRow } from './cards.js';
import { updateCostCard } from './cost.js';
import { updateDiagRow } from './diagnostics-row.js';
import { updateHealthCard } from './health.js';
import { hasKpiCards, kpiStore } from './store.js';
import { updateTelemetryCards } from './telemetry.js';

export { buildKpiRow, hasKpiCards };

function lastTrainLoss(state) {
  for (let i = state.metricHistory.length - 1; i >= 0; i--) {
    const m = state.metricHistory[i];
    if (m.train_loss > 0) return m;
  }
  return null;
}

export function updateKpis(podId) {
  const { cards } = kpiStore;
  if (!cards) return;
  updateBootPanel(podId);
  const state = getPodState(podId);
  const pod = app.pods.find((p) => p.id === podId) || {};
  const m = state.lastMetric;
  const running = pod.desiredStatus === 'RUNNING';

  updateTelemetryCards(state, pod);
  updateDiagRow(state);
  updateHealthCard(state);

  if (!m) {
    const waitMsg = state.finished ? 'Run finished — no metrics recorded'
      : running ? 'Waiting for first metric…' : 'No metrics recorded for this pod';
    cards.projected.note(waitMsg);
    cards.hero.note(waitMsg);
    cards.loss.note(waitMsg);
    updateCostCard(state, pod, null);
    updateRaceBanner(state, null, null, null);
    updateBaselineRow(state, null);
    return;
  }

  const msSeries = state.metricHistory.map((x) => x.step_avg_ms).filter((v) => v != null);
  const ema50 = d.ema(msSeries, d.ETA_EMA_N);
  const ema100 = d.ema(msSeries, d.AVG_EMA_N);
  const remaining = m.total_steps != null ? m.total_steps - m.step : null;
  const eta = state.finished ? 0 : d.etaMs(remaining, ema50);

  if (state.finished) {
    cards.projected.set({
      value: 'Finished',
      sub: state.exitCode === 0 ? 'Completed cleanly' : `Exit code ${state.exitCode ?? '?'}`,
      subClass: state.exitCode === 0 ? 'success' : 'danger',
    });
  } else if (eta != null) {
    cards.projected.set({ value: fmtDuration(eta), sub: `ETA ${fmtClock(new Date(Date.now() + eta))}` });
  } else {
    cards.projected.note('Waiting for step timing…');
  }
  if (cards.projected.spark) cards.projected.spark(msSeries);

  const batchTokens = state.info.batch_tokens ?? state.runRow?.batch_tokens ?? null;
  const finishBudgetMs = d.configFinishBudgetMs(state.runRow) ?? (m.train_time_ms ?? null);
  const race = d.finishRace({
    metric: m,
    history: state.metricHistory,
    baselineSamples: state.baselineSteps.map((s) => state.baselineByStep[s]),
    currentFinishBudgetMs: finishBudgetMs,
    baselineFinishBudgetMs: state.baselineFinishBudgetMs,
    batchTokens,
    emaMsPerStep: ema100,
    baselineSample: d.baselineAtStep(state.baselineByStep, state.baselineSteps, m.step),
  });

  if (race.state === 'ahead' || race.state === 'behind' || race.state === 'tied') {
    const ahead = race.state === 'ahead';
    const tied = race.state === 'tied';
    const tput = race.throughputTps != null ? `${fmtTps(race.throughputTps)} tok/s` : `${fmtMs(race.msPerStep)} ms/step`;
    const finishTxt = `${race.currentFinishLoss.toFixed(3)} vs ${race.baselineFinishLoss.toFixed(3)}`;
    cards.hero.set({
      value: tied ? '~0' : Math.abs(race.finishMarginLoss).toFixed(4),
      unit: tied ? 'tied' : (ahead ? 'ahead' : 'behind'),
      sub: `finish ${finishTxt} · ${tput}`,
      subClass: tied ? '' : (ahead ? 'success' : 'danger'),
      caption: tied ? 'Too close to call at budget' : (ahead ? 'Better finish at budget' : 'Worse finish at budget'),
      captionClass: tied ? '' : (ahead ? 'success' : 'danger'),
      accent: tied ? '' : (ahead ? 'success' : 'danger'),
    });
  } else if (race.state === 'unknown') {
    cards.hero.note('Waiting for enough loss trend to project the finish…');
  } else {
    cards.hero.note('No baseline selected — pick one below to start the race');
  }

  const lossD = d.deltaVsStepsAgo(state.metricHistory, 'train_loss', m.step, 100);
  if (lossD) {
    cards.loss.set({
      value: lossD.current.toFixed(4),
      sub: `${fmtDelta(lossD.delta, 4)} / ${fmtInt(lossD.stepsSpanned)} steps`,
      subClass: lossD.delta <= 0 ? 'success' : 'danger',
    });
  } else {
    const lt = lastTrainLoss(state);
    if (lt) cards.loss.set({ value: lt.train_loss.toFixed(4), sub: 'No earlier sample to compare yet' });
    else cards.loss.note('No training loss yet…');
  }

  updateCostCard(state, pod, eta);
  updateRaceBanner(state, m, race, eta);
  updateBaselineRow(state, race);
}
