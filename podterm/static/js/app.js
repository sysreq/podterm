// Entry module: init, tab routing, pod polling, action wiring.
import { app, on, dropPodState, getPodState, ensurePodStream, hydrateFromDb } from './state.js';
import { renderRunList, initRunList } from './runlist.js';
import { renderLiveView, initLive } from './live.js';
import { loadHistory, clearSelection, initHistory } from './history.js';
import { compareSelected, initCompare } from './compare.js';
import { openLaunchDialog, openImportDialog, initLaunch } from './launch.js';
import { initLogs } from './logs.js';
import { initConfigPanel } from './configpanel.js';
import { initDiagnostics } from './diagnostics.js';

// Browsers cap ~6 SSE connections per origin on HTTP/1.1 — stream at most
// this many pods at once (the active pod always gets one).
const MAX_STREAMS = 4;

function setStatus(text) {
  document.getElementById('status').textContent = text;
}

function setStatusSub(text) {
  document.getElementById('status-sub').textContent = text;
}

function setClusterPill(ok, text) {
  const pill = document.getElementById('cluster-pill');
  pill.classList.toggle('down', !ok);
  document.getElementById('cluster-pill-text').textContent = text;
}

// ── Tabs ──
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === tab));
  document.querySelectorAll('.tab-content').forEach((c) => c.classList.toggle('active', c.id === 'tab-' + tab));
  if (tab === 'history') loadHistory();
  if (tab === 'machines') renderMachines();
  // Plotly renders 0-width inside display:none containers — resize on reveal.
  const shown = document.getElementById('tab-' + tab);
  if (shown && window.Plotly) {
    shown.querySelectorAll('.js-plotly-plot').forEach((p) => Plotly.Plots.resize(p));
  }
}

// ── Machines (stub view) ──
function renderMachines() {
  const area = document.getElementById('machines-table-area');
  if (!app.pods.length) {
    area.innerHTML = '<div class="placeholder">No pods. Launch Run to create one.</div>';
    return;
  }
  let html = '<table><thead><tr><th>Name</th><th>ID</th><th>Status</th><th>$/hr</th><th>Image</th></tr></thead><tbody>';
  for (const p of app.pods) {
    html += `<tr><td>${p.name || '?'}</td><td>${p.id}</td><td>${p.desiredStatus || '?'}</td>
      <td>${p.costPerHr != null ? '$' + p.costPerHr : '?'}</td><td style="font-size:11px">${p.imageName || '?'}</td></tr>`;
  }
  area.innerHTML = html + '</tbody></table>';
}

// ── Pods ──
function connectRunningPods() {
  const running = app.pods.filter((p) => p.desiredStatus === 'RUNNING');
  // Active pod first so it always lands inside the stream cap.
  running.sort((a, b) => (a.id === app.activePod ? -1 : 0) - (b.id === app.activePod ? -1 : 0));
  let open = app.pods.filter((p) => getPodState(p.id).es).length;
  for (const p of running) {
    if (getPodState(p.id).es) continue;
    if (open >= MAX_STREAMS) break;
    ensurePodStream(p.id);
    hydrateFromDb(p.id);
    open++;
  }
}

async function refreshPods() {
  try {
    const r = await fetch('/api/pods');
    app.pods = await r.json();
    renderRunList(selectPod);
    connectRunningPods();
    const running = app.pods.filter((p) => p.desiredStatus === 'RUNNING').length;
    const queued = app.pods.length - running;
    setStatus(`${app.pods.length} pods, ${queued} queued`);
    setStatusSub('All systems nominal');
    setClusterPill(true, 'RunPod');
    if (document.getElementById('tab-machines').classList.contains('active')) renderMachines();
  } catch {
    setStatus('Cluster unreachable');
    setStatusSub('Check runpodctl and network, then Refresh');
    setClusterPill(false, 'RunPod — offline');
  }
}

function selectPod(podId) {
  app.activePod = podId;
  renderRunList(selectPod);
  switchTab('live');
  renderLiveView(podId);
}

async function stopPod() {
  if (!app.activePod) return;
  if (!confirm(`Stop run on pod ${app.activePod}? The pod will be terminated.`)) return;
  setStatus(`Stopping run on ${app.activePod}…`);
  try {
    await fetch(`/api/pods/${app.activePod}/stop`, { method: 'POST' });
    setStatus(`Run stopped (${app.activePod})`);
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

  initRunList();
  initLive();
  initLogs();
  initConfigPanel();
  initDiagnostics();
  initHistory();
  initCompare();
  initLaunch();

  refreshPods();
  setInterval(refreshPods, 15000);
}

init();
