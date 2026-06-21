// Live view orchestration: stream setup, charts, logs/config panels, and event wiring.
import { app, getPodState, ensurePodStream, hydrateFromDb, hydrateRunRow, on } from './state.js';
import { initLiveCharts, updateLiveCharts, updateBaselineTrace } from './charts.js';
import { renderLogs } from './logs.js';
import { renderConfigPanel } from './configpanel.js';
import { updateBootPanel } from './live/boot.js';
import { loadBaselineOptions, onBaselineChange, onPinBaselineClick } from './live/baseline.js';
import { buildKpiRow, hasKpiCards, updateKpis } from './live/kpis.js';

let dashboardPodId = null;

function renderLiveDashboard(podId) {
  const state = getPodState(podId);
  buildKpiRow();
  initLiveCharts();
  updateKpis(podId);
  updateLiveCharts(state);
  updateBaselineTrace(state);
  renderLogs(podId);
  renderConfigPanel(podId);
  loadBaselineOptions(podId);
  dashboardPodId = podId;
}

function ensureLiveDashboard(podId) {
  if (dashboardPodId === podId && hasKpiCards()) return;
  renderLiveDashboard(podId);
}

function syncLiveView(podId) {
  if (app.activePod !== podId) return false;
  const booting = updateBootPanel(podId);
  if (!booting) ensureLiveDashboard(podId);
  return !booting;
}

export function renderLiveView(podId) {
  ensurePodStream(podId);
  hydrateRunRow(podId);
  hydrateFromDb(podId);

  document.getElementById('live-placeholder').style.display = 'none';
  const booting = updateBootPanel(podId);
  document.getElementById('live-view').classList.add('visible');
  if (!booting) renderLiveDashboard(podId);
}

export function initLive() {
  document.getElementById('baseline-select').addEventListener('change', onBaselineChange);
  document.getElementById('baseline-pin').addEventListener('click', onPinBaselineClick);

  on('pod:metric', ({ podId }) => {
    if (!syncLiveView(podId) || !hasKpiCards()) return;
    updateLiveCharts(getPodState(podId));
    updateKpis(podId);
  });

  on('pod:boot', ({ podId }) => {
    syncLiveView(podId);
  });

  on('pod:log', ({ podId }) => {
    if (app.activePod !== podId) return;
    if (document.getElementById('live-view').classList.contains('booting')) syncLiveView(podId);
  });

  for (const evt of ['pod:memory', 'pod:info', 'pod:summary', 'pod:phase', 'pod:telemetry', 'pod:diagnostic', 'pod:health', 'pod:snapshot']) {
    on(evt, ({ podId }) => {
      if (!syncLiveView(podId)) return;
      updateKpis(podId);
    });
  }

  for (const evt of ['pod:reset', 'pod:hydrated', 'pod:run-row']) {
    on(evt, ({ podId }) => {
      if (!syncLiveView(podId) || !hasKpiCards()) return;
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
