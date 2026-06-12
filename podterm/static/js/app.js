// Entry module: init, tab routing, pod polling, action wiring.
import { app, on, dropPodState } from './state.js';
import { renderPodList } from './runlist.js';
import { renderLiveView, initLive } from './live.js';
import { loadHistory, clearSelection, initHistory } from './history.js';
import { compareSelected, initCompare } from './compare.js';
import { openLaunchDialog, openImportDialog, initLaunch } from './launch.js';

function setStatus(text) {
  document.getElementById('status').textContent = text;
}

// ── Tabs ──
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === tab));
  document.querySelectorAll('.tab-content').forEach((c) => c.classList.toggle('active', c.id === 'tab-' + tab));
  if (tab === 'history') loadHistory();
  // Plotly renders 0-width inside display:none containers — resize on reveal.
  const shown = document.getElementById('tab-' + tab);
  if (shown && window.Plotly) {
    shown.querySelectorAll('.js-plotly-plot').forEach((p) => Plotly.Plots.resize(p));
  }
}

// ── Pods ──
async function refreshPods() {
  try {
    const r = await fetch('/api/pods');
    app.pods = await r.json();
    renderPodList(selectPod);
    const running = app.pods.filter((p) => p.desiredStatus === 'RUNNING').length;
    setStatus(`${app.pods.length} pods, ${running} running`);
  } catch {
    setStatus('Error loading pods');
  }
}

function selectPod(podId) {
  app.activePod = podId;
  renderPodList(selectPod);
  switchTab('live');
  renderLiveView(podId);
}

async function stopPod() {
  if (!app.activePod) return;
  if (!confirm(`Terminate pod ${app.activePod}?`)) return;
  setStatus(`Stopping ${app.activePod}...`);
  try {
    await fetch(`/api/pods/${app.activePod}/stop`, { method: 'POST' });
    setStatus(`Terminated ${app.activePod}`);
    dropPodState(app.activePod);
    app.activePod = null;
    refreshPods();
  } catch (e) {
    setStatus(`Stop failed: ${e}`);
  }
}

// ── Init ──
function init() {
  document.querySelectorAll('.tab').forEach((t) => {
    t.addEventListener('click', () => switchTab(t.dataset.tab));
  });
  document.getElementById('btn-launch').addEventListener('click', openLaunchDialog);
  document.getElementById('btn-stop').addEventListener('click', stopPod);
  document.getElementById('btn-refresh').addEventListener('click', refreshPods);
  document.getElementById('btn-compare-selected').addEventListener('click', compareSelected);
  document.getElementById('btn-clear-selection').addEventListener('click', clearSelection);
  document.getElementById('btn-import-logs').addEventListener('click', openImportDialog);

  on('status', ({ text }) => setStatus(text));
  on('pods:refresh', refreshPods);
  on('pod:select', ({ podId }) => selectPod(podId));
  on('tab:switch', ({ tab }) => switchTab(tab));
  on('history:reload', loadHistory);

  initLive();
  initHistory();
  initCompare();
  initLaunch();

  refreshPods();
  setInterval(refreshPods, 15000);
}

init();
