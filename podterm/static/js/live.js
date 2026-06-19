// Live view orchestration: stream setup, charts, logs/config panels, and event wiring.
import { app, getPodState, ensurePodStream, hydrateFromDb, on } from './state.js';
import { initLiveCharts, updateLiveCharts, updateBaselineTrace } from './charts.js';
import { renderLogs } from './logs.js';
import { renderConfigPanel } from './configpanel.js';
import { updateBootPanel } from './live/boot.js';
import { loadBaselineOptions, onBaselineChange, onPinBaselineClick } from './live/baseline.js';
import { buildKpiRow, hasKpiCards, updateKpis } from './live/kpis.js';

export function renderLiveView(podId) {
  ensurePodStream(podId);
  hydrateFromDb(podId);
  const state = getPodState(podId);

  document.getElementById('live-placeholder').style.display = 'none';
  document.getElementById('live-view').classList.add('visible');
  buildKpiRow();
  initLiveCharts();
  updateKpis(podId);
  updateLiveCharts(state);
  updateBaselineTrace(state);
  renderLogs(podId);
  renderConfigPanel(podId);
  loadBaselineOptions(podId);
}

export function initLive() {
  document.getElementById('baseline-select').addEventListener('change', onBaselineChange);
  document.getElementById('baseline-pin').addEventListener('click', onPinBaselineClick);

  on('pod:metric', ({ podId }) => {
    if (app.activePod !== podId || !hasKpiCards()) return;
    updateLiveCharts(getPodState(podId));
    updateKpis(podId);
  });

  on('pod:boot', ({ podId }) => {
    if (app.activePod !== podId) return;
    updateBootPanel(podId);
  });

  for (const evt of ['pod:memory', 'pod:info', 'pod:summary', 'pod:phase', 'pod:telemetry', 'pod:diagnostic', 'pod:health']) {
    on(evt, ({ podId }) => {
      if (app.activePod !== podId) return;
      updateKpis(podId);
    });
  }

  for (const evt of ['pod:reset', 'pod:hydrated']) {
    on(evt, ({ podId }) => {
      if (app.activePod !== podId || !hasKpiCards()) return;
      const state = getPodState(podId);
      updateLiveCharts(state);
      updateBaselineTrace(state);
      updateKpis(podId);
    });
  }

  // Wall-clock cards (cost, ETA clock) tick even between metric events.
  setInterval(() => {
    if (app.activePod && hasKpiCards() && document.getElementById('live-view').classList.contains('visible')) {
      updateKpis(app.activePod);
    }
  }, 5000);
}
